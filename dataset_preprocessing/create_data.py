import sqlite3
import pandas as pd

pd.set_option('display.max_columns', None)
connection = sqlite3.connect('ParentReply.db')
c = connection.cursor()
limit = 100000
last_unix = 0
cur_length = limit
counter = 0

while cur_length == limit:

    df = pd.read_sql(
        "SELECT * FROM parent_reply LIMIT {}".format(limit), connection)
    last_unix = df.tail(1)['unix'].values[0]
    cur_length = len(df)

    with open('seq2seq/data1.txt', 'a', encoding='utf8') as f:
        for parent, comment in df[['parent', 'comment']].values:
            f.write(parent + '\t')
            f.write(comment + '\n')

    counter += 1
    if counter % 5 == 0:
        print(counter * limit, 'rows completed so far')
