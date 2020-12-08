import os
import pandas as pd
import numpy as np
def convertTrainLabel():
    df = pd.read_csv('../global-wheat-detection/train.csv')
    bboxs = np.stack(df['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
    for i, column in enumerate(['x', 'y', 'w', 'h']):
        df[column] = bboxs[:, i]
    df.drop(columns=['bbox'], inplace=True)
    df['x_center'] = df['x'] + df['w'] / 2
    df['y_center'] = df['y'] + df['h'] / 2
    df['classes'] = 0
    from tqdm.auto import tqdm
    import shutil as sh
    df = df[['image_id', 'x', 'y', 'w', 'h', 'x_center', 'y_center', 'classes']]

    index = list(set(df.image_id))

    source = 'train'
    if True:
        for fold in [0]:
            val_index = index[len(index) * fold // 100:len(index) * (fold + 1) // 100]
            for name, mini in tqdm(df.groupby('image_id')):
                if name in val_index:
                    path2save = 'val2017/'
                else:
                    path2save = 'train2017/'
                if not os.path.exists('convertor/fold{}/labels/'.format(fold) + path2save):
                    os.makedirs('convertor/fold{}/labels/'.format(fold) + path2save)
                with open('convertor/fold{}/labels/'.format(fold) + path2save + name + ".txt", 'w+') as f:
                    row = mini[['classes', 'x_center', 'y_center', 'w', 'h']].astype(float).values
                    row = row / 1024
                    row = row.astype(str)
                    for j in range(len(row)):
                        text = ' '.join(row[j])
                        f.write(text)
                        f.write("\n")
                if not os.path.exists('convertor/fold{}/images/{}'.format(fold, path2save)):
                    os.makedirs('convertor/fold{}/images/{}'.format(fold, path2save))
                sh.copy("../global-wheat-detection/{}/{}.jpg".format(source, name),
                        'convertor/fold{}/images/{}/{}.jpg'.format(fold, path2save, name))
convertTrainLabel()