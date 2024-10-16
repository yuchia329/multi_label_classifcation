import pandas as pd
data = []
for i in range(10):
    data.append(['aaa.cc', 'none', 'aaa.aa'])
df3 = pd.DataFrame(data, columns=["python", "sql", "java"])
df3.replace('none', '')
df3['CORE RELATIONS'] = df3[df3.columns[:]].apply(
    lambda x: ' '.join(x.dropna().astype(str).sort_values().loc[x != 'none']), axis=1)
# print(df3.columns[:-1])
df3.drop(columns=df3.columns[:3], axis=1, inplace=True)
df3.to_csv("./test.csv", index=True, index_label="ID")
x = range(10)
print(x)
