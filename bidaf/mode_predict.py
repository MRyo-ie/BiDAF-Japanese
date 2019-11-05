from . import BidirectionalAttentionFlow
from .scripts import MagnitudeVectors, get_best_span, get_word_char_loc_mapping



# 予測のテンプレート
def predict_ans(bidaf: BidirectionalAttentionFlow, mode, wordDB_dir,
                passage, question, squad_version=1.1, max_span_length=25, do_lowercase=True,
                return_char_loc=False, return_confidence_score=False):

    if mode == 'squad':
        from .tasks.squad import tokenize
    elif mode == 'ja_qa':
        # （未）
        from .tasks.ja_QA import TokenizerSP

    if type(passage) == list:
        assert all(type(pas) == str for pas in passage), "Input 'passage' must be of type 'string'"

        passage = [pas.strip() for pas in passage]
        contexts = []
        for pas in passage:
            context_tokens = tokenize(pas, do_lowercase)
            contexts.append(context_tokens)

        if do_lowercase:
            original_passage = [pas.lower() for pas in passage]
        else:
            original_passage = passage

    elif type(passage) == str:
        passage = passage.strip()
        context_tokens = tokenize(passage, do_lowercase)
        contexts = [context_tokens, ]

        if do_lowercase:
            original_passage = [passage.lower(), ]
        else:
            original_passage = [passage, ]

    else:
        raise TypeError("Input 'passage' must be either a 'string' or 'list of strings'")

    assert type(passage) == type(
        question), "Both 'passage' and 'question' must be either 'string' or a 'list of strings'"

    if type(question) == list:
        assert all(type(ques) == str for ques in question), "Input 'question' must be of type 'string'"
        assert len(passage) == len(
            question), "Both lists (passage and question) must contain same number of elements"

        questions = []
        for ques in question:
            question_tokens = tokenize(ques, do_lowercase)
            questions.append(question_tokens)

    elif type(question) == str:
        question_tokens = tokenize(question, do_lowercase)
        questions = [question_tokens, ]

    else:
        raise TypeError("Input 'question' must be either a 'string' or 'list of strings'")

    vectors = MagnitudeVectors(bidaf.emdim, wordDB_dir).load_vectors()
    context_batch = vectors.query(contexts, bidaf.max_passage_length)
    question_batch = vectors.query(questions, bidaf.max_query_length)

    y = bidaf.model.predict([context_batch, question_batch])
    y_pred_start = y[:, 0, :]
    y_pred_end = y[:, 1, :]

    # clearing the session releases memory by removing the model from memory
    # using this, you will need to load model every time before prediction
    # K.clear_session()

    batch_answer_span = []
    batch_confidence_score = []
    for sample_id in range(len(contexts)):
        answer_span, confidence_score = get_best_span(y_pred_start[sample_id, :], y_pred_end[sample_id, :],
                                                        len(contexts[sample_id]), squad_version, max_span_length)
        batch_answer_span.append(answer_span)
        batch_confidence_score.append(confidence_score)

    answers = []
    for index, answer_span in enumerate(batch_answer_span):
        context_tokens = contexts[index]
        start, end = answer_span[0], answer_span[1]

        # word index to character index mapping
        mapping = get_word_char_loc_mapping(original_passage[index], context_tokens)

        char_loc_start = mapping[start]
        # [1] => char_loc_end is set to point to one more character after the answer
        char_loc_end = mapping[end] + len(context_tokens[end])
        # [1] will help us getting a perfect slice without unnecessary increments/decrements
        ans = original_passage[index][char_loc_start:char_loc_end]

        return_dict = {
            "answer": ans,
        }

        if return_char_loc:
            return_dict["char_loc_start"] = char_loc_start
            return_dict["char_loc_end"] = char_loc_end - 1

        if return_confidence_score:
            return_dict["confidence_score"] = batch_confidence_score[index]

        answers.append(return_dict)

    if type(passage) == list:
        return answers
    else:
        return answers[0]

