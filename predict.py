from bidaf import BidirectionalAttentionFlow
from bidaf.mode_predict import predict_ans

# 初期化
bidaf_model = BidirectionalAttentionFlow(400)
bidaf_model.load_bidaf("bidaf/data/models/bidaf_50.h5")

print('\nWarming up を開始します！')
print(predict_ans(bidaf_model,
        "This is a tree", "What is this?"))
#=> {'answer': 'tree'}

print(predict_ans(bidaf_model,
        "This is a tree which highest in my country.", "What is this?"))
#=> {'answer': 'tree which highest in my country'}



if __name__ == "__main__":
    while True:
        print('\n\n')
        sentence = input('\n問題文 を入力（改行はしないでね！）\n>>  ')
        query = input('\n質問文 を入力\n>>  ')

        print('')
        ans = predict_ans(bidaf_model, sentence, query)
        print('\n\n回答： {}'.format(ans['answer']))

