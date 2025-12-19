
def build_strong_student_pipeline(M: int=2):
    # 学生强增流水线（多视图）：在 collate 中展开为 [M, ...]
    return [
        dict(type='Resize', scale=(1024, 768)),
        dict(type='RandomFlip', prob=0.5, direction='horizontal'),
        dict(type='RandAugment', policies=['AutoContrast', 'Equalize', 'Solarize', 'Color', 'Posterize', 'Contrast', 'Brightness', 'Sharpness'], num_policies=2, magnitude_level=7),
        dict(type='Cutout', n_holes=1, cutout_shape=(64,64)),
        dict(type='PackPoseInputs', meta_keys=('img_id','img_path','input_size','flip_indices','transformation_mat')),
    ]
