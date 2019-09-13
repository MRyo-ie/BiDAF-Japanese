from keras.layers import Input, TimeDistributed, LSTM, Bidirectional
from keras.models import Model, load_model
from keras.optimizers import Adadelta
from .layers import Highway, Similarity, C2QAttention, Q2CAttention, MergedContext, SpanBegin, SpanEnd, CombineOutputs
from .scripts import negative_avg_log_error, accuracy
from .scripts import ModelMGPU
from .scripts.utils import limit_GPU_memory_size


class BidirectionalAttentionFlow():

    def __init__(self, emdim, max_passage_length=None, max_query_length=None, num_highway_layers=2, num_decoders=1,
                 encoder_dropout=0, decoder_dropout=0):
        # GPU メモリの使用料を制限する
        #limit_GPU_memory_size(70)

        # モデルを構築
        self.emdim = emdim
        self.max_passage_length = max_passage_length
        self.max_query_length = max_query_length

        passage_input = Input(shape=(self.max_passage_length, emdim), dtype='float32', name="passage_input")
        question_input = Input(shape=(self.max_query_length, emdim), dtype='float32', name="question_input")

        question_embedding = question_input
        passage_embedding = passage_input
        for i in range(num_highway_layers):
            highway_layer = Highway(name='highway_{}'.format(i))
            question_layer = TimeDistributed(highway_layer, name=highway_layer.name + "_qtd")
            question_embedding = question_layer(question_embedding)
            passage_layer = TimeDistributed(highway_layer, name=highway_layer.name + "_ptd")
            passage_embedding = passage_layer(passage_embedding)

        encoder_layer = Bidirectional(LSTM(emdim, recurrent_dropout=encoder_dropout,
                                           return_sequences=True), name='bidirectional_encoder')
        encoded_question = encoder_layer(question_embedding)
        encoded_passage = encoder_layer(passage_embedding)

        similarity_matrix = Similarity(name='similarity_layer')([encoded_passage, encoded_question])

        context_to_query_attention = C2QAttention(name='context_to_query_attention')([
            similarity_matrix, encoded_question])
        query_to_context_attention = Q2CAttention(name='query_to_context_attention')([
            similarity_matrix, encoded_passage])

        merged_context = MergedContext(name='merged_context')(
            [encoded_passage, context_to_query_attention, query_to_context_attention])

        modeled_passage = merged_context
        for i in range(num_decoders):
            hidden_layer = Bidirectional(LSTM(emdim, recurrent_dropout=decoder_dropout,
                                              return_sequences=True), name='bidirectional_decoder_{}'.format(i))
            modeled_passage = hidden_layer(modeled_passage)

        span_begin_probabilities = SpanBegin(name='span_begin')([merged_context, modeled_passage])
        span_end_probabilities = SpanEnd(name='span_end')(
            [encoded_passage, merged_context, modeled_passage, span_begin_probabilities])

        output = CombineOutputs(name='combine_outputs')([span_begin_probabilities, span_end_probabilities])

        model = Model([passage_input, question_input], [output])

        model.summary()

        try:
            model = ModelMGPU(model)
        except:
            pass

        adadelta = Adadelta(lr=0.01)
        model.compile(loss=negative_avg_log_error, optimizer=adadelta, metrics=[accuracy])

        self.model = model

    def load_bidaf(self, path):
        custom_objects = {
            'Highway': Highway,
            'Similarity': Similarity,
            'C2QAttention': C2QAttention,
            'Q2CAttention': Q2CAttention,
            'MergedContext': MergedContext,
            'SpanBegin': SpanBegin,
            'SpanEnd': SpanEnd,
            'CombineOutputs': CombineOutputs,
            'negative_avg_log_error': negative_avg_log_error,
            'accuracy': accuracy
        }

        self.model = load_model(path, custom_objects=custom_objects)

