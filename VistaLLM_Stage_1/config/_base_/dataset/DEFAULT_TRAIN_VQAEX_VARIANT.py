VQAEX_TRAIN_COMMON_CFG = dict(
    type='VQAEXDataset',
    image_folder=r'REC/',
    template_file=r"VQA_CoT.json",
)

DEFAULT_TRAIN_VQAEX_VARIANT = dict(
    VQAE_train=dict(
        **VQAEX_TRAIN_COMMON_CFG,
        is_e_dataset=True,
        filename=r'vqa_E_train.jsonl',
    ),
    VQAX_train=dict(
        **VQAEX_TRAIN_COMMON_CFG,
        is_e_dataset=False,
        filename=r'vqa_X_train.jsonl',
    ),
)
