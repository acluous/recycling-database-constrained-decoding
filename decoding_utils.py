import torch
from tqdm import tqdm

def get_accuracy(preds, labels, labels_map):
  correct = 0
  for pred, label in zip(preds, labels):
    if pred in set(labels_map[label]):
      correct += 1
  return correct / len(preds)

def create_image_prompt(question, item=None):
  prompt = f"User:<image>{question}<end_of_utterance>\nAssistant:"
  if item is not None:
    prompt = prompt + f" {item}"
  return prompt
  
def create_database_ids(processor, database, query):
  database_ids = []
  for item in database:
    prompt = create_image_prompt(query, item)
    inputs = processor(text=prompt, return_tensors="pt")
    database_ids.append(inputs["input_ids"][0])
  return database_ids

@torch.no_grad
def database_constrained_decoding(model, batch, database, database_ids, eou_id=32002):
  input_ids = batch.pop("input_ids")
  assert input_ids.shape[0] == 1
  input_ids = input_ids[0]
  idx = input_ids.shape[0]

  item_ids = database_ids
  max_length = max([len(ids) for ids in item_ids])
  database_chosen = None

  for k in range(max_length - idx):
    outputs = model.forward(
        input_ids=input_ids[None, ...],
        pixel_values=batch["pixel_values"],
        pixel_attention_mask=batch["pixel_attention_mask"]
    )

    # filter for items in database that are the same as what we have so far
    database_valid, database_tokens = [], []
    for i, id in enumerate(item_ids):
      if torch.equal(id[:idx], input_ids) and id.shape[0] > idx:
        database_valid.append(i)
        database_tokens.append(id)

    if len(database_valid) == 0:
      return input_ids, database_chosen

    database_tokens = torch.tensor([id[idx] for id in database_tokens])
    sampled_idx = outputs.logits[:, -1, database_tokens].argmax(dim=-1)
    sampled_id = database_tokens[sampled_idx]

    # allow eou for early stop
    if database_chosen is not None:
      allow_eou = torch.equal(input_ids, item_ids[database_chosen])
      sampled_logit = outputs.logits[:, -1, sampled_id]
      eou_logit = outputs.logits[:, -1, eou_id]
      if eou_logit > sampled_logit and allow_eou:
        return input_ids, database_chosen

    database_chosen = database_valid[sampled_idx.item()]
    input_ids = torch.cat([input_ids, sampled_id], dim=0)
    idx += 1

  return input_ids, database_chosen

@torch.no_grad
def normalized_sequence_likelihood(model, batch, database, database_ids):
  database_logits = []
  for item_ids in database_ids:
    outputs = model.forward(
        input_ids=item_ids[None, ...],
        pixel_values=batch["pixel_values"],
        pixel_attention_mask=batch["pixel_attention_mask"]
    )
    # only consider tokens after the prompt
    idx = batch["input_ids"].shape[1]
    logits = outputs.logits[0, idx:, :]
    labels = item_ids[idx:]
    item_logits = logits[torch.arange(labels.shape[0]), labels]
    database_logits.append(item_logits.mean().item())
  database_logits = torch.tensor(database_logits)
  database_chosen = database_logits.argmax().item()
  input_ids = item_ids[database_chosen]
  return input_ids, database_chosen

@torch.no_grad
def mllm_classification(model, processor, image_dataset, database, query, mode="dcd"):
  preds = []
  all_database_chosen = []
  database_ids = create_database_ids(processor, database, query)
  for sample in tqdm(image_dataset):
    image = sample["image"]
    text = create_image_prompt(query)
    batch = processor(images=[image], text=[text], return_tensors="pt")
    if mode == "dcd":
      input_ids, database_chosen = database_constrained_decoding(model, batch, database, database_ids)
    elif mode == "nsl":
      input_ids, database_chosen = normalized_sequence_likelihood(model, batch, database, database_ids)
    else:
      raise NotImplementedError
    preds.append(database[database_chosen])
    all_database_chosen.append(database_chosen)
  labels = [sample["label"] for sample in image_dataset]
  return preds, labels, all_database_chosen

@torch.no_grad
def clip_embed_text(clip_model, clip_processor, database):
  all_text_features = []
  for text in tqdm(database):
    text = clip_processor(text=[text], return_tensors="pt")
    text_features = clip_model.get_text_features(**text)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    all_text_features.append(text_features)
  return torch.vstack(all_text_features)

@torch.no_grad
def clip_embed_image(clip_model, clip_processor, image_dataset):
  all_image_features = []
  for sample in tqdm(image_dataset):
    image = clip_processor(images=[sample["image"].convert("RGB")], return_tensors="pt")
    image_features = clip_model.get_image_features(**image)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    all_image_features.append(image_features)
  return torch.vstack(all_image_features)

@torch.no_grad
def contrastive_classification(clip_model, clip_processor, image_dataset, database):
  all_image_features = clip_embed_image(clip_model, clip_processor, image_dataset)
  all_text_features = clip_embed_text(clip_model, clip_processor, database)
  preds = []
  all_database_chosen = []
  for image in tqdm(all_image_features):
    sims = image @ all_text_features.T
    database_chosen = sims.argmax().item()
    preds.append(database[database_chosen])
    all_database_chosen.append(database_chosen)
  labels = [sample["label"] for sample in image_dataset]
  return preds, labels, all_database_chosen