# 文本预处理：清洗、分词、TF-IDF向量化

import re
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple, Optional


# 常见中文停用词
_STOP_WORDS: set = {
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
    '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有',
    '看', '好', '自己', '这', '他', '她', '它', '们', '那', '些', '所', '为',
    '所以', '因为', '但是', '然而', '而且', '虽然', '如果', '可以', '这个',
    '那个', '什么', '怎么', '哪', '吗', '啊', '哦', '吧', '呢', '嘛', '哈',
    '进行', '通过', '根据', '按照', '关于', '对于', '以及', '或者', '并且',
    '其中', '其他', '以上', '以下', '之前', '之后', '目前', '已经', '还',
    '将', '把', '被', '让', '向', '从', '以', '之', '中', '等', '等等',
    '需要', '可能', '应该', '必须', '能够', '不能', '不同', '主要', '包括',
    '使用', '利用', '采用', '具有', '存在', '发生', '产生', '发展', '提出',
}


def clean_text(text: str) -> str:
    # 去URL、邮箱、特殊字符，只留中文和字母数字
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ''
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^一-龥a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def jieba_cut(text: str) -> List[str]:
    # jieba精确模式分词，没装就回退到简单切分
    if not text or len(text.strip()) == 0:
        return []
    try:
        import jieba
        words = jieba.lcut(text)
    except ImportError:
        print('[警告] jieba 未安装，使用简单字符切分。建议: pip install jieba')
        text_clean = re.sub(r'\s+', '', text)
        words = []
        for i in range(len(text_clean)):
            if i < len(text_clean) - 1:
                words.append(text_clean[i:i + 2])
            words.append(text_clean[i])
    return words


def filter_stopwords(words: List[str],
                     custom_stopwords: Optional[set] = None) -> List[str]:
    # 去停用词、短词、纯数字
    if not words:
        return []
    stopwords = _STOP_WORDS.copy()
    if custom_stopwords:
        stopwords.update(custom_stopwords)
    result = []
    for w in words:
        w = w.strip()
        if (len(w) >= 2
                and not w.isdigit()
                and not (len(w) == 1 and w.isascii())
                and w not in stopwords):
            result.append(w)
    return result


def preprocess_document(text: str) -> List[str]:
    # 完整预处理流水线：清洗 → 分词 → 去停用词
    cleaned = clean_text(text)
    words = jieba_cut(cleaned)
    filtered = filter_stopwords(words)
    return filtered


# ---------- TF-IDF ----------

def compute_tf(doc_words: List[str]) -> Dict[str, float]:
    # 词频 TF = 该词出现次数 / 文档总词数
    if not doc_words:
        return {}
    total = len(doc_words)
    counter = Counter(doc_words)
    return {word: count / total for word, count in counter.items()}


def compute_idf(documents: List[List[str]],
                smooth: bool = True) -> Dict[str, float]:
    # 逆文档频率 IDF = log(总文档数 / (出现该词的文档数 + 1))
    if not documents:
        return {}
    n_docs = len(documents)
    doc_freq: Dict[str, int] = {}
    for doc in documents:
        unique_words = set(doc)
        for word in unique_words:
            doc_freq[word] = doc_freq.get(word, 0) + 1
    if smooth:
        return {w: np.log(n_docs / (freq + 1)) for w, freq in doc_freq.items()}
    else:
        return {w: np.log(n_docs / freq) if freq > 0 else 0.0
                for w, freq in doc_freq.items()}


def tfidf_vectorize(documents: List[List[str]]) -> Tuple[np.ndarray, List[str]]:
    # 对文档集做TF-IDF向量化，返回 (特征矩阵, 词表)
    n_docs = len(documents)
    if n_docs == 0:
        return np.array([]).reshape(0, 0), []
    idf = compute_idf(documents)
    vocab = sorted(idf.keys(), key=lambda w: idf[w], reverse=True)
    n_features = len(vocab)
    if n_features == 0:
        return np.zeros((n_docs, 0)), []
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    matrix = np.zeros((n_docs, n_features), dtype=np.float64)
    for j, doc in enumerate(documents):
        tf = compute_tf(doc)
        for word, tf_val in tf.items():
            if word in word_to_idx:
                i = word_to_idx[word]
                matrix[j, i] = tf_val * idf[word]
    return matrix, vocab


# ---------- 模拟数据 ----------

def simulate_multiformat_files() -> List[Dict[str, str]]:
    # 模拟多格式文件（Word/PDF/Excel/图片OCR/纯文本）
    templates = [
        {
            'filename': '采购合同_2024Q1.docx',
            'format': 'Word',
            'content': (
                '采购合同 甲方科技有限公司与乙方供应链公司就2024年第一季度'
                '办公设备采购达成协议 合同金额人民币伍拾万元整 交货日期为2024年3月31日'
                '违约责任条款包括逾期交货每日按合同金额千分之五支付违约金'
                '质量保证期限为验收合格后十二个月 争议解决方式为仲裁'
            )
        },
        {
            'filename': '技术服务协议_2024.pdf',
            'format': 'PDF',
            'content': (
                '技术服务协议 甲方委托乙方提供信息系统运维服务 服务期限自2024年1月1日'
                '至2024年12月31日 服务费用每月人民币贰万元整 服务内容包括服务器监控'
                '数据库维护 安全漏洞修复 乙方需保证系统可用性不低于百分之九十九点九'
                '保密条款约定乙方不得泄露甲方任何业务数据和技术资料'
            )
        },
        {
            'filename': '季度经营分析报告_Q3.xlsx',
            'format': 'Excel',
            'content': (
                '第三季度经营分析报告 营业收入较上季度增长百分之十五 净利润率达到百分之二十三'
                '市场占有率提升至百分之十八点五 新客户获取成本下降百分之十二'
                '核心产品线销售额突破两千万元 研发投入占营收比例百分之八'
                '下一季度重点推进数字化转型项目和海外市场拓展计划'
            )
        },
        {
            'filename': '安全生产检查报告.jpg',
            'format': '图片(OCR)',
            'content': (
                '安全生产检查报告 检查日期2024年6月15日 检查区域生产车间A区和仓库B区'
                '发现安全隐患三项 消防通道堵塞 电气线路老化 安全标识缺失'
                '整改期限为十五个工作日 整改责任人张三 复查日期2024年7月5日'
                '要求各部门立即开展安全隐患自查自纠工作确保安全生产零事故'
            )
        },
        {
            'filename': '数据安全管理办法_2024.pdf',
            'format': 'PDF',
            'content': (
                '数据安全管理办法 依据中华人民共和国数据安全法和个人信息保护法制定'
                '数据分类分级管理 核心业务数据定为三级敏感数据 财务数据和客户信息定为二级'
                '数据访问实行最小权限原则 所有数据操作需留日志记录保存不少于六个月'
                '数据跨境传输需经安全评估和审批 违反规定将追究相关责任'
            )
        },
        {
            'filename': '合规审查通知.docx',
            'format': 'Word',
            'content': (
                '合规审查通知 根据监管机构要求开展年度合规审查工作 审查范围覆盖所有业务部门'
                '重点审查反洗钱制度执行 客户信息保护 内部控制有效性三个方面'
                '各部门需在2024年7月31日前提交自查报告 合规部门将于8月进行现场检查'
                '发现问题需在十个工作日内完成整改并提交整改报告'
            )
        },
        {
            'filename': '年度预算方案_2025.xlsx',
            'format': 'Excel',
            'content': (
                '2025年度预算方案 总预算规模为人民币八千万元 其中人员成本占比百分之四十五'
                '运营成本占比百分之三十 研发投入占比百分之十五 市场推广占比百分之十'
                '各部门预算需在11月30日前提交 财务部将在12月进行汇总审核'
                '预算执行率将纳入各部门年度绩效考核指标'
            )
        },
        {
            'filename': '差旅报销凭证_20240815.jpg',
            'format': '图片(OCR)',
            'content': (
                '差旅报销凭证 报销人李四 出差日期2024年8月10日至8月14日 出差地点北京市'
                '交通费用高铁往返票共计壹仟贰佰元 住宿费用四晚共计贰仟元'
                '餐饮补助每日壹佰元共计伍佰元 合计报销金额叁仟柒佰元整'
            )
        },
        {
            'filename': '员工绩效考核表_2024H1.docx',
            'format': 'Word',
            'content': (
                '员工绩效考核表 考核周期2024年上半年 被考核人王五 岗位软件开发工程师'
                '工作业绩完成核心模块开发任务按时交付率百分之九十五 代码质量评分A级'
                '团队协作能力优秀 主动承担技术攻关任务 获得客户书面表扬两次'
                '需改进方面技术文档编写规范性有待提高 建议参加项目管理培训'
            )
        },
        {
            'filename': '培训需求调研报告.txt',
            'format': '纯文本',
            'content': (
                '培训需求调研报告 调研对象全体员工共三百五十人 回收有效问卷三百二十份'
                '最受欢迎培训主题前五位 数字化技能提升 项目管理 沟通技巧 数据分析 领导力'
                '百分之七十八的员工倾向于线上培训方式 期望每季度至少一次专业培训'
                '建议建立内部讲师制度 鼓励技术骨干分享实践经验'
            )
        },
    ]
    return templates


if __name__ == '__main__':
    test_text = '2024年第三季度经营分析报告显示，营业收入较上季度增长15%！'
    print('原始文本:', test_text)
    cleaned = clean_text(test_text)
    print('清洗后:', cleaned)
    words = jieba_cut(cleaned)
    print('分词结果:', words)
    filtered = filter_stopwords(words)
    print('去停用词:', filtered)

    files = simulate_multiformat_files()
    print(f'\n模拟文件总数: {len(files)}')
    for f in files[:3]:
        print(f"  [{f['format']}] {f['filename']}: {f['content'][:40]}...")

    docs = [preprocess_document(f['content']) for f in files]
    matrix, vocab = tfidf_vectorize(docs)
    print(f'\nTF-IDF 矩阵形状: {matrix.shape}')
    print(f'词汇量: {len(vocab)}')
