import numpy as np

def evaluation_prediction(real_y, pred_y):
    """
    only support 1d y temporarily

    :param real_y: 1d array, (n, )
    :param pred_y: 1d array, (n, )
    :return: common metric like accuracy, precision, recall, F1
    """
    real_y = np.array(real_y)
    pred_y = np.array(pred_y)
    assert np.shape(real_y) == np.shape(pred_y)
    result = {}
    labels = set(real_y) | set(pred_y)

    # precision, recall, F1
    for l in labels:
        if l not in result:
            result[l] = {}
        real_P = float(sum(real_y == l))
        pred_P = float(sum(pred_y == l))
        TP = float(sum(pred_y[real_y == pred_y] == l))
        FP = float(sum(pred_y[real_y != pred_y] == l))
        FN = float(sum(pred_y[real_y != pred_y] != l))
        TN = float(sum(pred_y[real_y == pred_y] != l))

        result[l]['precision'] = TP / max(pred_P, 1)
        result[l]['recall'] = TP / max(real_P, 1)
        result[l]['F1'] = 2 * TP / max((pred_P + real_P), 1)
        result[l]['FPR'] = FP / max((FP + TN), 1)
        result[l]['support'] = real_P

    # accuracy
    true_cases = float(sum(real_y == pred_y))
    all_cases = len(real_y)
    result['overall accuracy'] = true_cases / max(all_cases, 1)
    result['class average accuracy'] = float(np.mean([result[l]['recall'] for l in labels]))

    return result


if __name__ == '__main__':
    from pprint import pprint

    real_y = ['123', 'asdf']
    pred_y = ['123', 'wef']
    pprint(evaluation_prediction(real_y, pred_y))

    real_y = [1, 0]
    pred_y = [1.0, 1]
    pprint(evaluation_prediction(real_y, pred_y))