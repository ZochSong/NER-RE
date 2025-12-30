from named_entity_recogition import BertCrfForNer, NerWorker
from relation_extract import BertForRe, ReWorker
import torch
import os


class NerReModel:
    def __init__(self,
                 ner_model_path='./ner_model_save/',
                 re_model_path='./re_model_save/',
                 device=None,
                 use_crf=True):
        """
        Args:
            ner_model_path: NER模型保存路径
            re_model_path: RE模型保存路径
            device: 设备 (cuda/cpu)
            use_crf: NER是否使用CRF模型
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_crf = use_crf

        print("初始化 NER + RE Model...")
        print(f"设备: {self.device}")

        # 加载NER模型
        self._load_ner_model(ner_model_path)

        # 加载RE模型
        self._load_re_model(re_model_path)

        print("Model 初始化完成!\n")

    def _load_ner_model(self, model_path):
        print("\n加载NER模型...")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"NER模型路径不存在: {model_path}")

        # 加载配置获取标签数量
        import joblib
        config_path = os.path.join(model_path, 'config.pkl')
        if os.path.exists(config_path):
            config = joblib.load(config_path)
            num_labels = config['num_labels']
        else:
            raise FileNotFoundError(f"未找到NER配置文件: {config_path}")

        # 创建模型
        if self.use_crf:
            ner_model = BertCrfForNer(num_labels=num_labels)
            print("   使用 BERT+CRF 模型")
        else:
            from ner import BertForNer
            ner_model = BertForNer(num_labels=num_labels)
            print("   使用 BERT 模型")

        # 创建worker并加载权重
        self.ner_worker = NerWorker(ner_model, device=self.device)
        self.ner_worker.load(model_path)

    def _load_re_model(self, model_path):
        print("\n加载RE模型...")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RE模型路径不存在: {model_path}")

        # 加载配置获取关系数量
        import joblib
        config_path = os.path.join(model_path, 're_config.pkl')
        if os.path.exists(config_path):
            config = joblib.load(config_path)
            # relation2idx 已经包含了所有关系（包括UNK），直接使用其长度
            num_relations = len(config['relation2idx'])
            print(f"   关系数量: {num_relations}")
            print(f"   关系类型: {list(config['relation2idx'].keys())}")
        else:
            raise FileNotFoundError(f"未找到RE配置文件: {config_path}")

        re_model = BertForRe(num_relations=num_relations)

        self.re_worker = ReWorker(re_model, device=self.device)
        self.re_worker.load(model_path)

    def extract_entities(self, text):
        """
        从文本中提取实体

        Args:
            text: 输入文本

        Returns:
            实体列表: [(entity_text, entity_type), ...]
        """
        token_tags = self.ner_worker.predict(text)

        # 将BIO标签转换为实体
        entities = []
        current_entity = []
        current_type = None

        for token, tag in token_tags:
            if tag.startswith('B-'):
                # 新实体开始
                if current_entity:
                    entity_text = ''.join(current_entity)
                    entities.append((entity_text, current_type))
                current_entity = [token]
                current_type = tag[2:]  # 去掉 'B-'
            elif tag.startswith('I-'):
            # elif tag.startswith('I-') and current_type == tag[2:]:
                # 继续当前实体
                current_entity.append(token)
                current_type = tag[2:]
            else:
                # 非实体或实体结束
                if current_entity:
                    entity_text = ''.join(current_entity)
                    entities.append((entity_text, current_type))
                current_entity = []
                current_type = None

        # 处理最后一个实体
        if current_entity:
            entity_text = ''.join(current_entity)
            entities.append((entity_text, current_type))

        return entities

    def extract_relations(self, text, entities, threshold=0.6):
        """
        从实体列表中提取关系

        Args:
            text: 原始文本
            entities: 实体列表 [(entity_text, entity_type), ...]
            threshold: 置信度阈值

        Returns:
            关系列表: [(entity1, relation, entity2), ...]
        """
        return self.re_worker.extract_relations_from_entities(
            entities=entities,
            sentence=text,
            threshold=threshold
        )

    def extract(self, text, relation_threshold=0.6, verbose=False):
        """
        端到端知识抽取

        Args:
            text: 输入文本
            relation_threshold: 关系抽取的置信度阈值
            verbose: 是否显示详细信息

        Returns:
            dict: {
                'text': 原文本,
                'entities': [(entity, type), ...],
                'relations': [(entity1, relation, entity2), ...]
            }
        """
        if verbose:
            print(f"\n输入文本: {text}")
            print("=" * 60)

        # Step 1: 提取实体
        entities = self.extract_entities(text)

        if verbose:
            print(f"\n识别到 {len(entities)} 个实体:")
            for entity, entity_type in entities:
                print(f"   - {entity} ({entity_type})")

        # Step 2: 提取关系
        relations = []
        if len(entities) > 1:
            relations = self.extract_relations(text, entities, threshold=relation_threshold)

            if verbose:
                print(f"\n识别到 {len(relations)} 个关系:")
                for e1, rel, e2 in relations:
                    print(f"   - {e1} --[{rel}]--> {e2}")
        elif verbose:
            print(f"\n实体数量不足，无法提取关系")

        return {
            'text': text,
            'entities': entities,
            'relations': relations
        }

    def batch_extract(self, texts, relation_threshold=0.6, verbose=False):
        """
        批量提取知识

        Args:
            texts: 文本列表
            relation_threshold: 关系抽取的置信度阈值
            verbose: 是否显示详细信息

        Returns:
            list: 每个文本的提取结果
        """
        results = []
        for i, text in enumerate(texts):
            if verbose:
                print(f"\n{'='*60}")
                print(f"处理文本 {i+1}/{len(texts)}")
                print(f"{'='*60}")

            result = self.extract(text, relation_threshold=relation_threshold, verbose=verbose)
            results.append(result)

        return results

    def visualize_result(self, result):
        raise NotImplementedError("可视化功能待实现")