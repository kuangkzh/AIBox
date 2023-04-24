from datasets import load_from_disk, concatenate_datasets, Dataset


def build_dataset():
    d1 = load_from_disk("/home/nlper_data/kuangzh/datasets/liyucheng/zhihu_26k")['train']
    # d2 = load_from_disk("/home/nlper_data/kuangzh/datasets/liyucheng/zhihu_rlhf_3k")['train']
    d3 = load_from_disk("/home/nlper_data/kuangzh/datasets/suolyer/zhihu")
    d3 = concatenate_datasets([d3['validation'], d3['test']])
    d4 = load_from_disk("/home/nlper_data/kuangzh/datasets/wangrui6/Zhihu-KOL")['train']

    data = {
        "title": d1['INSTRUCTION'] + d3['title'] + d4['INSTRUCTION'],
        "response": d1['RESPONSE'] + d3['content'] + d4['RESPONSE'],
    }
    return Dataset.from_dict(data)
