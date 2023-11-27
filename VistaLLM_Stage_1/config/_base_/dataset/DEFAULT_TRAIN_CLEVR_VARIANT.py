CLEVR_TRAIN_COMMON_CFG = dict(
    type='ClevrDataset',
    filename=r'CLEVR_train_questions_with_ans.jsonl',
    image_folder=r'CLEVR/CLEVR_v1.0/images/train',
    scene_graph_file=r"CLEVR_train_scenes.jsonl",
)

DEFAULT_TRAIN_CLEVR_VARIANT = dict(
    CLEVR_A=dict(
        **CLEVR_TRAIN_COMMON_CFG,
        version='q-a',
        template_file=r"VQA.json",
    ),
    CLEVR_S=dict(
        **CLEVR_TRAIN_COMMON_CFG,
        version='q-s',
        template_file=r"VQA_CoT.json",
    ),
    CLEVR_BS=dict(
        **CLEVR_TRAIN_COMMON_CFG,
        version='q-bs',
        template_file=r"VQA_PCoT.json",
    ),
)
