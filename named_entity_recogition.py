from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import BertModel, BertTokenizer

import joblib
import numpy as np
import os
import torch


class CRF(nn.Module):
    """
    用于学习标签之间的转移规律，保证标签序列合法
    """
    def __init__(self, num_tags, batch_first=True):
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first

        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def forward(self, emissions, tags, mask=None, reduction='mean'):
        """
        计算负对数似然损失
        Args:
            emissions: [batch, seq_len, num_tags] - 发射分数
            tags: [batch, seq_len] - 真实标签
            mask: [batch, seq_len] - 掩码 (1=真实, 0=padding)
            reduction: 'mean' or 'sum'
        Returns:
            loss: 标量损失
        """
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        # 计算真实路径分数
        numerator = self._compute_score(emissions, tags, mask)

        # 计算所有路径分数(log-sum-exp)
        denominator = self._compute_normalizer(emissions, mask)

        # 负对数似然
        llh = denominator - numerator

        if reduction == 'mean':
            return llh.mean()
        elif reduction == 'sum':
            return llh.sum()
        else:
            return llh

    def _compute_score(self, emissions, tags, mask):
        """计算真实路径的分数"""
        _, seq_length = tags.size()
        mask = mask.float()

        # 初始分数
        score = emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)

        # 累加发射分数和转移分数
        for i in range(1, seq_length):
            # 发射分数
            emit_score = emissions[:, i].gather(1, tags[:, i].unsqueeze(1)).squeeze(1)

            # 转移分数: from tags[i-1] to tags[i]
            trans_score = self.transitions[tags[:, i], tags[:, i - 1]]

            # 只在mask=1的位置累加
            score = score + (emit_score + trans_score) * mask[:, i]

        return score

    def _compute_normalizer(self, emissions, mask):
        """前向算法:计算所有可能路径的log-sum-exp"""
        _, seq_length, _ = emissions.size()
        mask = mask.float()

        # 初始化alpha: [batch, num_tags]
        alpha = emissions[:, 0]

        # 前向传播
        for i in range(1, seq_length):
            # alpha: [batch, num_tags, 1]
            # emissions: [batch, 1, num_tags]
            # transitions: [num_tags, num_tags]

            emit_score = emissions[:, i].unsqueeze(1)  # [batch, 1, num_tags]
            trans_score = self.transitions.unsqueeze(0)  # [1, num_tags, num_tags]
            next_alpha = alpha.unsqueeze(2) + emit_score + trans_score  # [batch, num_tags, num_tags]

            # log-sum-exp
            next_alpha = torch.logsumexp(next_alpha, dim=1)  # [batch, num_tags]

            # 根据mask更新
            alpha = next_alpha * mask[:, i].unsqueeze(1) + alpha * (1 - mask[:, i].unsqueeze(1))

        # 最终分数
        return torch.logsumexp(alpha, dim=1)

    def decode(self, emissions, mask=None):
        """
        维特比解码:找到最优标签序列
        Args:
            emissions: [batch, seq_len, num_tags]
            mask: [batch, seq_len]
        Returns:
            List[List[int]]: 最优标签序列
        """
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.uint8, device=emissions.device)

        return self._viterbi_decode(emissions, mask)

    def _viterbi_decode(self, emissions, mask):
        """维特比算法"""
        batch_size, seq_length, _ = emissions.size()

        # 初始化
        score = emissions[:, 0]  # [batch, num_tags]
        history = []

        # 前向
        for i in range(1, seq_length):
            # score: [batch, num_tags, 1]
            # transitions: [num_tags, num_tags]
            # emissions: [batch, 1, num_tags]

            broadcast_score = score.unsqueeze(2)  # [batch, num_tags, 1]
            broadcast_emission = emissions[:, i].unsqueeze(1)  # [batch, 1, num_tags]
            next_score = broadcast_score + self.transitions.unsqueeze(0) + broadcast_emission

            # 找最大值和索引
            next_score, indices = next_score.max(dim=1)  # [batch, num_tags]

            # 应用mask
            score = torch.where(mask[:, i].unsqueeze(1).bool(), next_score, score)
            history.append(indices)

        # 回溯
        best_tags_list = []
        for idx in range(batch_size):
            # 找到最后的最优标签
            seq_len = mask[idx].sum().item()
            best_last_tag = score[idx].argmax().item()

            best_tags = [best_last_tag]

            # 回溯
            for hist in reversed(history[:int(seq_len) - 1]):
                best_last_tag = hist[idx, best_tags[-1]].item()
                best_tags.append(best_last_tag)

            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list


def load_ner_data(file, encoding='utf-8'):
    sentences = []
    labels = []
    sentence = []
    label = []

    with open(file, 'r', encoding=encoding) as f:
        for line in f:
            if line.strip():
                word, tag = line.strip().split() 
                sentence.append(word)
                label.append(tag)
            else:
                if sentence and label:
                    sentences.append(sentence)
                    labels.append(label)
                sentence = []
                label = []
        if sentence and label:
            sentences.append(sentence)
            labels.append(label)
    return sentences, labels

def encode_sentences(sentences, labels, tokenizer, max_len=128):
    """
    编码句子和标签，确保标签与tokenizer输出对齐
    """
    input_ids = []
    attention_masks = []
    label_ids = []

    # 创建标签映射（确保顺序稳定）
    label2idx = {label: i for i, label in enumerate(sorted(set(sum(labels, []))))}
    label2idx["PAD"] = len(label2idx)

    # 为特殊token添加标签（[CLS] 和 [SEP] 使用 PAD 标签）
    pad_label_id = label2idx["PAD"]

    for sentence, label in zip(sentences, labels):
        inputs = tokenizer(
            sentence,
            is_split_into_words=True,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids.append(inputs['input_ids'])
        attention_masks.append(inputs['attention_mask'])

        # 构建标签序列：[CLS] + labels + [SEP] + [PAD]...
        # 注意：需要为 [CLS] 和 [SEP] 添加 PAD 标签
        label_sequence = [pad_label_id]  # [CLS] 的标签

        # 添加实际标签
        for tag in label[:max_len - 2]:  # 留出 [CLS] 和 [SEP] 的位置
            label_sequence.append(label2idx[tag])

        # 添加 [SEP] 的标签
        label_sequence.append(pad_label_id)

        # 填充到 max_len
        while len(label_sequence) < max_len:
            label_sequence.append(pad_label_id)

        # 截断到 max_len（以防万一）
        label_sequence = label_sequence[:max_len]

        label_ids.append(torch.tensor(label_sequence, dtype=torch.long))

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    label_ids = torch.stack(label_ids)

    assert input_ids.size(0) > 0, "No input IDs to load."
    assert attention_masks.size(0) > 0, "No attention masks to load."
    assert label_ids.size(0) > 0, "No label IDs to load."

    print("Data successfully loaded and encoded.")
    print(f"  Label mapping: {label2idx}")
    print(f"  Sample shapes - input_ids: {input_ids.shape}, labels: {label_ids.shape}")

    return input_ids, attention_masks, label_ids, label2idx


class BertForNer(nn.Module):
    """原始的 BERT + Linear 模型（用于对比）"""
    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels

        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.dropout = nn.Dropout(0.3)
        self.decoder = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0])
        logits = self.decoder(sequence_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.num_labels - 1)
            loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return loss, logits
        return logits


class BertCrfForNer(nn.Module):
    """
    BERT + CRF 模型
    相比纯BERT模型，CRF层可以学习标签之间的转移规律，确保输出序列合法
    例如：避免 O -> I-PER 这种非法转移
    """
    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels

        # BERT 编码器
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.dropout = nn.Dropout(0.5)  # CRF模型可以用更高的dropout

        # 发射层：将BERT输出映射到标签空间
        self.hidden2tag = nn.Linear(self.bert.config.hidden_size, num_labels)

        # CRF 层
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, seq_len] (可选)
        Returns:
            训练模式：返回 (loss, emissions)
            推理模式：返回 best_paths（维特比解码结果）
        """
        # BERT 编码
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0])  # [batch, seq_len, hidden_size]

        # 发射分数
        emissions = self.hidden2tag(sequence_output)  # [batch, seq_len, num_labels]

        if labels is not None:
            # 训练模式：计算CRF损失
            loss = self.crf(emissions, labels, mask=attention_mask.byte())
            return loss, emissions
        else:
            # 推理模式：维特比解码
            best_paths = self.crf.decode(emissions, mask=attention_mask.byte())
            return best_paths


class NerWorker(nn.Module):
    def __init__(self, model, optimizer=torch.optim.AdamW, device=torch.device('cuda'), batch_size=32):
        super().__init__()
        self.model = model
        self.optimizer = optimizer(self.model.parameters(), lr=5e-5)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.device = device
        self.batch_size = batch_size
        self.label2idx = None
        self.training_data_loader = None
        self.test_data_loader = None

    def load_training_data(self, file):
        sentences, labels = load_ner_data(file)
        input_ids, attention_masks,\
            label_ids, self.label2idx = encode_sentences(sentences, labels, tokenizer=self.tokenizer)
        
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
            
            preds.extend(torch.argmax(logits, dim=2).cpu().numpy())
            true_labels.extend(b_labels.cpu().numpy())
            
        preds = [item for sublist in preds for item in sublist]
        true_labels = [item for sublist in true_labels for item in sublist]
        print(classification_report(true_labels, preds, labels=list(self.label2idx.values()), target_names=list(self.label2idx.keys())))

    def predict(self, sentence):
        """
        预测单个句子的实体标签
        自动检测模型类型（BERT 或 BERT+CRF）并使用相应的解码方式
        """
        self.model.eval()
        self.model.to(self.device)

        inputs = self.tokenizer.encode_plus(
            sentence,
            return_tensors='pt',
            max_length=128,
            truncation=True,
            padding='max_length'
        )
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            output = self.model(input_ids, attention_mask)

        # 检测模型类型并相应处理
        is_crf_model = isinstance(self.model, BertCrfForNer)

        if is_crf_model:
            # CRF 模型：output 是 List[List[int]]（维特比解码结果）
            pred_tags = output[0]  # 取第一个（也是唯一一个）样本的预测
            tokens = self.tokenizer.tokenize(sentence)

            # pred_tags 已经是最优路径，直接转换为标签名
            idx2label = {v: k for k, v in self.label2idx.items()}
            tags = [idx2label[tag_id] for tag_id in pred_tags[1:len(tokens)+1]]  # 跳过[CLS]

        else:
            # 普通 BERT 模型：output 是 logits
            logits = output
            if isinstance(logits, tuple):
                logits = logits[0]

            logits = logits[0].cpu().numpy()  # [seq_len, num_labels]
            attention_mask_np = attention_mask[0].cpu().numpy()

            # 只考虑attention_mask为1的logits
            valid_logits = logits[attention_mask_np == 1]

            tokens = self.tokenizer.tokenize(sentence)
            tags = [list(self.label2idx.keys())[i] for i in np.argmax(valid_logits[1:-1], axis=1)]

        # 返回每个字对应的标签
        return list(zip(tokens[:len(tags)], tags[:len(tokens)]))

    def save(self, save_path='./model_save/'):
        """保存模型、优化器状态和标签映射"""
        os.makedirs(save_path, exist_ok=True)

        # 保存模型权重
        model_save_path_pth = os.path.join(save_path, 'ner_bert.pth')
        torch.save(self.model.state_dict(), model_save_path_pth)

        # 保存优化器状态
        optimizer_save_path_pth = os.path.join(save_path, 'ner_optimizer.pth')
        torch.save(self.optimizer.state_dict(), optimizer_save_path_pth)

        # 保存label2idx映射和配置
        config_path = os.path.join(save_path, 'config.pkl')
        config = {
            'label2idx': self.label2idx,
            'num_labels': self.model.num_labels,
            'batch_size': self.batch_size
        }
        joblib.dump(config, config_path)

        print(f"✅ Model saved successfully to {save_path}")
        print(f"   - Model weights: ner_bert.pth")
        print(f"   - Optimizer state: ner_optimizer.pth")
        print(f"   - Config (label2idx): config.pkl")

    def load(self, save_path='./ner_model_save/'):
        """加载模型、优化器状态和标签映射"""
        # 检查文件是否存在
        model_path = os.path.join(save_path, 'ner_bert.pth')
        optimizer_path = os.path.join(save_path, 'ner_optimizer.pth')
        config_path = os.path.join(save_path, 'config.pkl')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model file not found: {model_path}")

        # 加载配置和label2idx
        if os.path.exists(config_path):
            config = joblib.load(config_path)
            self.label2idx = config['label2idx']
            print(f"✅ Loaded label2idx with {len(self.label2idx)} labels: {list(self.label2idx.keys())}")
        else:
            print(f"⚠️  Warning: config.pkl not found. label2idx may not be initialized.")

        # 加载模型权重
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        print(f"✅ Model weights loaded from {model_path}")

        # 加载优化器状态（可选，主要用于继续训练）
        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
            print(f"✅ Optimizer state loaded from {optimizer_path}")
        else:
            print(f"⚠️  Optimizer state not found (OK for inference only)")

        print(f"✅ Model loaded successfully from {save_path}")
