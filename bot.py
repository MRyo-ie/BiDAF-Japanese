from bidaf import BidirectionalAttentionFlow
bidaf_model = BidirectionalAttentionFlow(400)
bidaf_model.load_bidaf("bidaf/data/models/bidaf_50.h5")

print('\nWarming up を開始します！')
print(bidaf_model.predict_ans("This is a tree", "What is this?"))
#=> {'answer': 'tree'}

print(bidaf_model.predict_ans("This is a tree which highest in my country.", "What is this?"))
#=> {'answer': 'tree which highest in my country'}



if __name__ == "__main__":
    while True:
        print('\n\n')
        sentence = input('\n問題文 を入力（改行はしないでね！）\n>>  ')
        query = input('\n質問文 を入力\n>>  ')

        print('')
        ans = bidaf_model.predict_ans(sentence, query)
        print('\n\n回答： {}'.format(ans['answer']))

