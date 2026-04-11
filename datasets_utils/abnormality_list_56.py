abnormality_dict = {
    'Pulmonary arteries':[
        'Enlarged pulmonary artery',          # 0
        'Pulmonary embolism',                 # 1
        'Acute pulmonary embolism',           # 2
        'Chronic pulmonary embolism',         # 3
        'Main pulmonary artery PE',           # 4
        'Lobar pulmonary artery PE',          # 5
        'Segmental pulmonary artery PE',      # 6 v25
        'Subsegmental pulmonary artery PE',   # 7 v25
        'Saddle pulmonary embolism',          # 8 v25
        'Pulmonary artery aneurysm',          # 9 v25
    ],

    'Lungs and Airways':[
        # Parenchymal lesions
        'Lung opacity',                       # 10
        'Pulmonary consolidation',            # 11
        'Atelectasis',                        # 12
        'Lung mass',                          # 13 v25
        'Lung nodule',                        # 14
        'Cavitary lesion',                    # 15 v25
        'Pulmonary fibrotic sequela',         # 16
        'Organizing pneumonia',               # 17 v25
        'Pulmonary edema',                    # 18 v25
        'Mosaic attenuation pattern',         # 19
        'Interlobular septal thickening',     # 20
        # Airway lesions
        'Peribronchial thickening',           # 21
        'Bronchial wall thickening',          # 22
        'Bronchiectasis',                     # 23
        'Emphysema',                          # 24
    ],

    'Pleura':[
        'Pleural effusion',                   # 25
        'Pneumothorax',                       # 26
        'Pleural thickening',                 # 27 v25
        'Pleural plaque',                     # 28 v25
        'Loculated pleural effusion',         # 29 v25
        'Pleural calcification',              # 30 v25
        'Hydropneumothorax',                  # 31 v25
    ],

    'Heart':[
        'Enlarged ascending aorta',           # 32
        'Cardiomegaly',                       # 33
        'Coronary artery calcification',      # 34
        'Right heart strain',                 # 35
        'Pericardial effusion',               # 36
        'Left atrial enlargement',            # 37 v25
        'Ventricular hypertrophy',            # 38 v25
        'Aortic aneurysm / dissection',       # 39 v25
        'Pulmonary venous congestion',        # 40 v25
    ],

    'Mediastinum and Hila':[
        'Mediastinal lymphadenopathy',        # 41
        'Hilar lymphadenopathy',              # 42
        'Esophagus abnormality',              # 43
        'Hiatal hernia',                      # 44
        'Atherosclerotic calcification',      # 45
        'Mediastinal mass',                   # 46 v25
    ],

    'Chest Wall and Lower Neck':[
        'Chest wall mass',                    # 47
        'Subcutaneous lesion',                # 48 v25
        'Thyroid nodule',                     # 49
        'Enlarged thyroid',                   # 50
        'Lymph node enlargement',             # 51 v25
    ],

    'Chest Bones':[
        'Acute fracture',                     # 52
        'Suspicious osseous lesion',          # 53
        'Sclerotic osseous lesion',           # 54 v25
        'Lytic osseous lesion',               # 55 v25
    ],
}

abnormality_list = [item for group in abnormality_dict.values() for item in group]

finding_section_list = list(abnormality_dict.keys())

    
