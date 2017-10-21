from spotlight.cross_validation import random_train_test_split
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.datasets.synthetic import generate_sequential
from spotlight.evaluation import rmse_score
from spotlight.evaluation import sequence_mrr_score
from spotlight.factorization.explicit import ExplicitFactorizationModel
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.sequence.implicit import ImplicitSequenceModel


dataset = get_movielens_dataset(variant='100K')
train, test = random_train_test_split(dataset)


def train_and_test(model, train, test, score):
    print('Train and test {}'.format(model))
    model.fit(train, verbose=True)

    _score = score(model, test)
    print('score({}): {}'.format(score, _score))


explicit_model = ExplicitFactorizationModel(n_iter=1)
train_and_test(explicit_model, train, test, rmse_score)

implicit_model = ImplicitFactorizationModel(n_iter=3,
                                            loss='bpr')
train_and_test(implicit_model, train, test, rmse_score)

train = train.to_sequence()
test = test.to_sequence()

implicit_cnn_model = ImplicitSequenceModel(n_iter=3,
                                           representation='cnn',
                                           loss='bpr')
train_and_test(implicit_cnn_model, train, test, sequence_mrr_score)
