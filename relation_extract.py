from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import BertModel, BertTokenizer

import joblib
import numpy as np
import os
import torch


def load_re_data(file, encoding='utf-8'):
    entity_pairs = []
    sentences = []
    relations = []
    with open(file, 'r', encoding=encoding) as f:
        for line in f:
            # 跳过空行和注释行
            if line.strip() and not line.strip().startswith('#'):
                parts = line.strip().split('\t')
                if len(parts) == 4:
                    entity1, entity2, sentence, relation = parts
                    entity_pairs.append((entity1, entity2))
                    sentences.append(sentence)
                    relations.append(relation)
    return entity_pairs, sentences, relations

def encode_sentences_4re(entity_pairs, sentences, relations, tokenizer, max_len=128):
    input_ids = []
    attention_masks = []
    relation2idx = {relation: i for i, relation in enumerate(set(relations))}
    relation2idx['UNK'] = len(relation2idx)

    for sentence, (entity1, entity2) in zip(sentences, entity_pairs):
        # 标记实体位置 - 改进版，避免重复替换
        # 找到实体第一次出现的位置
        e1_pos = sentence.find(entity1)
        e2_pos = sentence.find(entity2)

        # 按位置顺序标记（从后往前，避免索引偏移）
        if e1_pos != -1 and e2_pos != -1:
            if e1_pos < e2_pos:
                # entity1在前
                marked_sentence = (sentence[:e2_pos] +
                                 '[E2]' + entity2 + '[/E2]' +
                                 sentence[e2_pos + len(entity2):])
                e1_new_pos = marked_sentence.find(entity1)
                marked_sentence = (marked_sentence[:e1_new_pos] +
                                 '[E1]' + entity1 + '[/E1]' +
                                 marked_sentence[e1_new_pos + len(entity1):])
            else:
                # entity2在前
                marked_sentence = (sentence[:e1_pos] +
                                 '[E1]' + entity1 + '[/E1]' +
                                 sentence[e1_pos + len(entity1):])
                e2_new_pos = marked_sentence.find(entity2)
                marked_sentence = (marked_sentence[:e2_new_pos] +
                                 '[E2]' + entity2 + '[/E2]' +
                                 marked_sentence[e2_new_pos + len(entity2):])
        elif e1_pos != -1:
            # 只找到entity1
            marked_sentence = sentence.replace(entity1, '[E1]' + entity1 + '[/E1]', 1)
        elif e2_pos != -1:
            # 只找到entity2
            marked_sentence = sentence.replace(entity2, '[E2]' + entity2 + '[/E2]', 1)
        else:
            # 都没找到，保持原样
            marked_sentence = sentence

        inputs = tokenizer.encode_plus(marked_sentence,
                                       add_special_tokens=True,
                                       max_length=max_len,
                                       padding='max_length',
                                       truncation=True,
                                       return_attention_mask=True,
                                       return_tensors='pt')
        
        input_ids.append(inputs['input_ids'])
        attention_masks.append(inputs['attention_mask'])
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    relation_ids = torch.tensor([relation2idx[relation] for relation in relations], dtype=torch.long)
    assert input_ids.size(0) > 0, "No input IDs to load."
    assert attention_masks.size(0) > 0, "No attention masks to load."
    assert relation_ids.size(0) > 0, "No relation IDs to load."
    print("Data successfully loaded and encoded.")
    return input_ids, attention_masks, relation_ids, relation2idx


class BertForRe(nn.Module):
    def __init__(self, num_relations):
        super().__init__()
        self.num_relations = num_relations
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_relations)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1] # [CLS] token的输出
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        # logits = self.classifier(pooled_output) ** 2

        if labels is not None:
            class_weights = torch.ones(self.num_relations, device=labels.device)
            class_weights[self.num_relations - 1] = 2
            loss_fct = nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fct(logits, labels)
            return loss, logits
        return logits


class ReWorker(nn.Module):
    def __init__(self, model, optimizer=torch.optim.AdamW, device=torch.device('cuda'), batch_size=32):
        super().__init__()
        self.model = model
        self.optimizer = optimizer(self.model.parameters(), lr=5e-5)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.device = device
        self.batch_size = batch_size
        self.relation2idx = None
        self.training_data_loader = None
        self.test_data_loader = None

    def load_training_data(self, file):
        entity_pairs, sentences, relations = load_re_data(file)
        input_ids, attention_masks,\
            label_ids, self.relation2idx = encode_sentences_4re(entity_pairs, sentences, relations, tokenizer=self.tokenizer)
        
        data = TensorDataset(input_ids, attention_masks, label_ids)
        sampler = RandomSampler(data)
        self.training_data_loader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)
    
    def train(self, epochs):
        self.model.train()
        self.model.to(self.device)
        
        for epoch in range(epochs):
            total_loss = 0
            for step, batch in enumerate(self.training_data_loader):
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_attention_mask, b_labels = batch
                
                self.model.zero_grad()
                
                loss, logits = self.model(b_input_ids, b_attention_mask, b_labels)
                total_loss += loss.item()
                
                loss.backward()
                self.optimizer.step()

            avg_loss = total_loss / len(self.training_data_loader)
            print(f"Epoch {epoch + 1}, Loss: {avg_loss}")
    
    def evaluate(self):
        self.model.eval()
        preds, true_labels = [], []

        for batch in self.test_data_loader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_attention_mask, b_labels = batch

            with torch.no_grad():
                logits = self.model(b_input_ids, b_attention_mask)
            
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(b_labels.cpu().numpy())
            
        preds = [item for sublist in preds for item in sublist]
        true_labels = [item for sublist in true_labels for item in sublist]
        print(classification_report(true_labels, preds, labels=list(self.relation2idx.values()), target_names=list(self.relation2idx.keys())))

    def predict(self, entity1, entity2, sentence, threshold=0.6, moreInfo=False, strict=True):
        """
        预测两个实体之间的关系
        Args:
            entity1: 第一个实体
            entity2: 第二个实体
            sentence: 包含这两个实体的句子
            threshold: 置信度阈值
            moreInfo: 是否打印详细信息
            strict: 是否严格模式（暂未使用）
        Returns:
            关系类型字符串
        """
        self.model.eval()
        self.model.to(self.device)

        # 标记实体位置 - 改进版，避免重复替换
        e1_pos = sentence.find(entity1)
        e2_pos = sentence.find(entity2)

        # 按位置顺序标记（从后往前，避免索引偏移）
        if e1_pos != -1 and e2_pos != -1:
            if e1_pos < e2_pos:
                # entity1在前
                marked_sentence = (sentence[:e2_pos] +
                                 '[E2]' + entity2 + '[/E2]' +
                                 sentence[e2_pos + len(entity2):])
                e1_new_pos = marked_sentence.find(entity1)
                marked_sentence = (marked_sentence[:e1_new_pos] +
                                 '[E1]' + entity1 + '[/E1]' +
                                 marked_sentence[e1_new_pos + len(entity1):])
            else:
                # entity2在前
                marked_sentence = (sentence[:e1_pos] +
                                 '[E1]' + entity1 + '[/E1]' +
                                 sentence[e1_pos + len(entity1):])
                e2_new_pos = marked_sentence.find(entity2)
                marked_sentence = (marked_sentence[:e2_new_pos] +
                                 '[E2]' + entity2 + '[/E2]' +
                                 marked_sentence[e2_new_pos + len(entity2):])
        elif e1_pos != -1:
            # 只找到entity1
            marked_sentence = sentence.replace(entity1, '[E1]' + entity1 + '[/E1]', 1)
        elif e2_pos != -1:
            # 只找到entity2
            marked_sentence = sentence.replace(entity2, '[E2]' + entity2 + '[/E2]', 1)
        else:
            # 都没找到，返回UNK
            if moreInfo:
                print(f"Warning: 实体 '{entity1}' 或 '{entity2}' 未在句子中找到")
            return 'UNK'

        inputs = self.tokenizer.encode_plus(marked_sentence,
                                            return_tensors='pt',
                                            max_length=128,
                                            truncation=True,
                                            padding='max_length')

        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)

        probs = torch.softmax(logits, dim=1)

        max_prob, relation_id = torch.max(probs, dim=1)

        if moreInfo:
            print(f"Marked sentence: {marked_sentence}")
            print(f"Relation mapping: {self.relation2idx}")
            print(f'Probabilities: {probs.cpu().numpy()}')
            print(f'Max probability: {max_prob.item():.4f}')

        if max_prob.item() < threshold:
            return 'UNK'

        relation = list(self.relation2idx.keys())[relation_id.item()]
        return relation

    def extract_relations_from_entities(self, entities, sentence, threshold=0.6):
        """
        从实体列表中自动提取所有可能的关系
        Args:
            entities: 实体列表，格式为 [(entity1, type1), (entity2, type2), ...]
                     或者简单的 [entity1, entity2, ...]
            sentence: 原句子
            threshold: 置信度阈值
        Returns:
            关系三元组列表: [(entity1, relation, entity2), ...]
        """
        relations = []
        enrolled_pairs = []
        
        # 处理输入格式
        if entities and isinstance(entities[0], tuple):
            entity_names = [e[0] for e in entities]
        else:
            entity_names = entities

        for i, e1 in enumerate(entity_names):
            for j, e2 in enumerate(entity_names):
                if i != j and not (e1 in e2 or e2 in e1):
                    if {e1, e2} in enrolled_pairs:
                        continue
                    enrolled_pairs.append(frozenset({e1, e2}))
                    relation = self.predict(e1, e2, sentence, threshold=threshold)
                    if relation != 'UNK':
                        relations.append((e1, relation, e2))

        return relations

    def save(self, save_path='./re_model_save/'):
        """保存模型、优化器和关系映射"""
        os.makedirs(save_path, exist_ok=True)
        model_save_path_pth = os.path.join(save_path, 're_bert.pth')
        optimizer_save_path_pth = os.path.join(save_path, 're_optimizer.pth')
        config_path = os.path.join(save_path, 're_config.pkl')

        # 保存模型和优化器
        torch.save(self.model.state_dict(), model_save_path_pth)
        torch.save(self.optimizer.state_dict(), optimizer_save_path_pth)

        # 保存关系映射和配置
        config = {
            'relation2idx': self.relation2idx,
            'batch_size': self.batch_size
        }
        joblib.dump(config, config_path)

        print(f"✅ Model saved successfully to {save_path}")
        print(f"   - Model weights: re_bert.pth")
        print(f"   - Optimizer state: re_optimizer.pth")
        print(f"   - Config (relation2idx): re_config.pkl")
        print(f"   - Relations: {list(self.relation2idx.keys())}")

    def load(self, save_path='./re_model_save/'):
        """加载模型、优化器和关系映射"""
        model_path = os.path.join(save_path, 're_bert.pth')
        optimizer_path = os.path.join(save_path, 're_optimizer.pth')
        config_path = os.path.join(save_path, 're_config.pkl')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model file not found: {model_path}")

        # 加载配置和relation2idx
        if os.path.exists(config_path):
            config = joblib.load(config_path)
            self.relation2idx = config['relation2idx']
            print(f"✅ Loaded relation2idx with {len(self.relation2idx)} relations: {list(self.relation2idx.keys())}")
        else:
            print(f"⚠️  Warning: re_config.pkl not found. relation2idx may not be initialized.")

        # 加载模型权重
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        print(f"✅ Model weights loaded from {model_path}")

        # 加载优化器状态（可选）
        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
            print(f"✅ Optimizer state loaded from {optimizer_path}")
        else:
            print(f"⚠️  Optimizer state not found (OK for inference only)")

        print(f"✅ Model loaded successfully from {save_path}")

