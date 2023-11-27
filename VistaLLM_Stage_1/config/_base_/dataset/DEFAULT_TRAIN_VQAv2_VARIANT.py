VQAv2_TRAIN_COMMON_CFG = dict(
    type='VQAv2Dataset',
    filename=r'v2_OpenEnded_mscoco_train2014_questions.jsonl',
    image_folder=r'VQAv2/combined',
    template_file=r"VQA.json",
)

DEFAULT_TRAIN_VQAv2_VARIANT = dict(
    VQAv2_train=dict(**VQAv2_TRAIN_COMMON_CFG),
)