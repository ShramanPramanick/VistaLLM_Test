POPE_TEST_COMMON_CFG = dict(
    type='POPEVQADataset',
    image_folder=r'val2014',
)

DEFAULT_TEST_POPE_VARIANT = dict(
    COCO_POPE_RANDOM_q_a=dict(
        **POPE_TEST_COMMON_CFG,
        filename='coco_pope_random.jsonl',
        template_file=r'VQA.json'
    ),
    COCO_POPE_RANDOM_q_bca=dict(
        **POPE_TEST_COMMON_CFG,
        filename='coco_pope_random.jsonl',
        template_file=r'VQA_BCoT_ChatGPT.json'
    ),
    COCO_POPE_POPULAR_q_a=dict(
        **POPE_TEST_COMMON_CFG,
        filename='coco_pope_popular.jsonl',
        template_file=r'VQA.json'
    ),
    COCO_POPE_POPULAR_q_bca=dict(
        **POPE_TEST_COMMON_CFG,
        filename='coco_pope_popular.jsonl',
        template_file=r'VQA_BCoT_ChatGPT.json'
    ),
    COCO_POPE_ADVERSARIAL_q_a=dict(
        **POPE_TEST_COMMON_CFG,
        filename='coco_pope_adversarial.jsonl',
        template_file=r'VQA.json'
    ),
    COCO_POPE_ADVERSARIAL_q_bca=dict(
        **POPE_TEST_COMMON_CFG,
        filename='coco_pope_adversarial.jsonl',
        template_file=r'VQA_BCoT_ChatGPT.json'
    ),
)