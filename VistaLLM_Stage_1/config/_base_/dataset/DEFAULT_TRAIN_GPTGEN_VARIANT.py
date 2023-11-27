GPTGEN_TRAIN_COMMON_CFG = dict(
    type='GPT4Gen',
    filename=r'GPT4GEN_BoxCoT_train.jsonl',
    image_folder=r'/datasets01/flickr30k-images',
)

DEFAULT_TRAIN_GPTGEN_VARIANT = dict(
    GPT4GEN_QA=dict(**GPTGEN_TRAIN_COMMON_CFG, version='a', template_file=r"VQA.json"),
    GPT4GEN_QC=dict(**GPTGEN_TRAIN_COMMON_CFG, version='c', template_file=r"template/VQA_CoT.json"),
    GPT4GEN_QBC=dict(**GPTGEN_TRAIN_COMMON_CFG, version='bc', template_file=r"VQA_BCoT_ChatGPT.json"),

    GPT4GEN_RD_QBC=dict(
        type=GPTGEN_TRAIN_COMMON_CFG['type'],
        image_folder=GPTGEN_TRAIN_COMMON_CFG['image_folder'],
        filename='GPT4GEN_RD_BoxCoT_train.jsonl',
        version='bc',
        template_file=r"VQA_BCoT_ChatGPT.json"),
)
