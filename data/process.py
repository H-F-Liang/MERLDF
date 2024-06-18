import pandas as pd

dataset = ['dxy', 'muzhi']  # ['dxy', 'muzhi', 'mz4', 'mz10']

for i in range(len(dataset)):
    sym_set = set()
    dis_set = set()
    df = pd.read_csv(f'./{dataset[i]}/train_set.csv', sep='|')
    data = []
    for row_data in df.iterrows():
        row = {
            'explicit_symptom': '',
            'implicit_symptom': '',
            'label': row_data[1]['label']
        }
        dis_set.add(row_data[1]['label'])
        for j, s in enumerate(str(row_data[1]['explicit_symptom']).split('，')):
            if s != 'nan':
                sym_set.add(s)
                row['explicit_symptom'] += s + ' True '
        for j, s in enumerate(str(row_data[1]['implicit_symptom']).split('，')):
            if s != 'nan':
                sym_set.add(s)
                row['implicit_symptom'] += s + ' False '
        for j, s in enumerate(str(row_data[1]['true_symptom']).split('，')):
            if s != 'nan':
                sym_set.add(s)
                row['implicit_symptom'] += ' ' + s + ' True '
        for j, s in enumerate(str(row_data[1]['none_symptom']).split('，')):
            if s != 'nan':
                sym_set.add(s)
                row['implicit_symptom'] += ' ' + s + ' False '
        data.append(row)
    pd.DataFrame(data).to_csv(f'./{dataset[i]}/train_cn.txt', sep=';', index=False, header=False)

    df = pd.read_csv(f'./{dataset[i]}/test_set.csv', sep='|')
    data = []
    for row_data in df.iterrows():
        row = {
            'explicit_symptom': '',
            'implicit_symptom': '',
            'label': row_data[1]['label']
        }
        dis_set.add(row_data[1]['label'])
        for j, s in enumerate(str(row_data[1]['explicit_symptom']).split('，')):
            if s != 'nan':
                sym_set.add(s)
                row['explicit_symptom'] += s + ' True '
        for j, s in enumerate(str(row_data[1]['implicit_symptom']).split('，')):
            if s != 'nan':
                sym_set.add(s)
                row['implicit_symptom'] += s + ' False '
        for j, s in enumerate(str(row_data[1]['true_symptom']).split('，')):
            if s != 'nan':
                sym_set.add(s)
                row['implicit_symptom'] += ' ' + s + ' True '
        for j, s in enumerate(str(row_data[1]['none_symptom']).split('，')):
            if s != 'nan':
                sym_set.add(s)
                row['implicit_symptom'] += ' ' + s + ' False '
        data.append(row)
    pd.DataFrame(data).to_csv(f'./{dataset[i]}/test_cn.txt', sep=';', index=False, header=False)

    pd.DataFrame(list(sym_set)).to_csv(f'./{dataset[i]}/symptoms.txt', sep='\t', index=False, header=False)
    pd.DataFrame(list(dis_set)).to_csv(f'./{dataset[i]}/diseases.txt', sep='\t', index=False, header=False)