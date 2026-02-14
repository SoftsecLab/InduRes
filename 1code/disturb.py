import random
import torch
import os
import stanza
import json
from transformers import BertTokenizer, BertForMaskedLM
from tqdm import tqdm

# ========== 初始化 Stanza ==========
MODEL_DIR = '/root/autodl-tmp/9/923/default'
nlp = stanza.Pipeline(
    lang='zh',
    processors='tokenize,pos,lemma,depparse',
    dir=MODEL_DIR,
    download_method=None,
    tokenize_model_path=f"{MODEL_DIR}/tokenize/gsdsimp.pt",
    pos_model_path=f"{MODEL_DIR}/pos/gsdsimp.pt",
    depparse_model_path=f"{MODEL_DIR}/depparse/gsdsimp.pt",
    verbose=False
)


# ========== 依存关系扰动 ==========
def dependency_based_perturbation(text, debug=False):
    doc = nlp(text)
    if not doc.sentences:
        return text
    words = doc.sentences[0].words
    word_list = [w.text for w in words]
    rules = ["advmod", "obj", "amod"]

    def apply_rule(rule):
        perturbed = word_list.copy()
        if rule == "advmod":
            advs = [w.text for w in words if w.deprel == "advmod"]
            if advs:
                adv = random.choice(advs)
                perturbed.remove(adv)
                perturbed.insert(0, adv) if random.random() < 0.5 else perturbed.append(adv)
        elif rule == "obj":
            objs = [w.text for w in words if w.deprel == "obj"]
            if objs:
                obj = random.choice(objs)
                perturbed.remove(obj)
                perturbed.insert(0, obj) if random.random() < 0.5 else perturbed.append(obj)
        elif rule == "amod":
            mods = [w.text for w in words if w.deprel in ["amod", "nummod"]]
            if mods:
                mod = random.choice(mods)
                perturbed.remove(mod)
                perturbed.insert(0, mod) if random.random() < 0.5 else perturbed.append(mod)
        return ''.join(perturbed)

    tried = []
    perturbed_text = text
    for _ in range(3):
        remaining = [r for r in rules if r not in tried]
        if not remaining:
            break
        rule = random.choice(remaining)
        tried.append(rule)
        perturbed_text = apply_rule(rule)
        if perturbed_text != text:
            return perturbed_text

    return text


# ========== BERT 多词掩码 + top-k ==========
def bert_multi_mask_replacement(text, model, tokenizer, num_masks=2, top_k=10, temperature=1.0):
    model.eval()
    input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True).cuda()
    length = input_ids.size(1)
    mask_count = min(num_masks, max(1, length - 2))
    masked_positions = random.sample(range(1, length - 1), mask_count)
    perturbed_ids = input_ids.clone()
    for pos in masked_positions:
        perturbed_ids[0, pos] = tokenizer.mask_token_id
    with torch.no_grad():
        outputs = model(perturbed_ids)
        logits = outputs.logits
    for pos in masked_positions:
        logits_pos = logits[0, pos] / temperature
        probs = torch.softmax(logits_pos, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, k=top_k)
        sampled_id = random.choices(topk_indices.tolist(), weights=topk_probs.tolist(), k=1)[0]
        perturbed_ids[0, pos] = sampled_id
    perturbed_text = tokenizer.decode(perturbed_ids[0], skip_special_tokens=True)
    return perturbed_text



def random_swap(text):
    chars = list(text)
    if len(chars) > 1:
        i, j = random.sample(range(len(chars)), 2)
        chars[i], chars[j] = chars[j], chars[i]
    return ''.join(chars)


def random_deletion(text):
    chars = list(text)
    if len(chars) > 1:
        chars.pop(random.randrange(len(chars)))
    return ''.join(chars)


def random_insertion(text):
    chars = list(text)
    if not chars:
        return text
    insert_char = random.choice(chars)
    pos = random.randint(0, len(chars))
    chars.insert(pos, insert_char)
    return ''.join(chars)


# ========== 综合扰动 ==========
def perturb_texts_extended(texts, model, tokenizer, typo_dict=None, strength=1):
    perturbed_texts = []
    pool_light = ["bert_multi_mask", "dependency_based", "spelling_error", "random_swap", "random_insertion"]
    pool_heavy = pool_light + ["sentence_reconstruction", "random_deletion"]

    for text in texts:
        perturbed_text = text
        if strength == 1:
            perturb_type_list = random.sample(pool_light, 1)
        elif strength == 2:
            perturb_type_list = random.sample(pool_light, 2)
        else:
            perturb_type_list = random.sample(pool_heavy, 3)

        for perturb_type in perturb_type_list:
            if perturb_type == "bert_multi_mask":
                perturbed_text = bert_multi_mask_replacement(perturbed_text, model, tokenizer,
                                                             num_masks=2, top_k=10, temperature=1.2)
            elif perturb_type == "dependency_based":
                perturbed_text = dependency_based_perturbation(perturbed_text)
            elif perturb_type == "spelling_error" and typo_dict:
                for k, v in typo_dict.items():
                    if k in perturbed_text:
                        perturbed_text = perturbed_text.replace(k, random.choice(v), 1)
                        break
            elif perturb_type == "random_swap":
                perturbed_text = random_swap(perturbed_text)
            elif perturb_type == "random_insertion":
                perturbed_text = random_insertion(perturbed_text)
            elif perturb_type == "random_deletion":
                perturbed_text = random_deletion(perturbed_text)
            elif perturb_type == "sentence_reconstruction":
                perturbed_text = ''.join(random.sample(list(perturbed_text), len(perturbed_text)))

        perturbed_texts.append(perturbed_text)

    return perturbed_texts


# ========== 主程序：对 JSON 文件进行扰动 ==========
if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('/root/autodl-tmp/9/923/bert-base-chinese')
    model = BertForMaskedLM.from_pretrained('/root/autodl-tmp/9/923/bert-base-chinese').cuda()

    with open("/root/autodl-tmp/9/923/final_typo_dict.json", "r", encoding="utf-8") as f:
        typo_dict = json.load(f)

    input_path = input("请输入待扰动的 JSON 文件路径：").strip()
    output_path = input("请输入输出文件路径：").strip()
    strength = int(input("请输入扰动强度 (1=轻度, 2=中度, 3=重度): ").strip())

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [item["text"] for item in data]
    perturbed_texts = perturb_texts_extended(texts, model, tokenizer, typo_dict=typo_dict, strength=strength)

    new_data = []
    for original, new_item in zip(data, perturbed_texts):
        new_entry = original.copy()
        new_entry["text"] = new_item
        new_data.append(new_entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 扰动完成！结果已保存至 {output_path}")
