import numpy as np


def distribute_batch(numbers, rows, columns):
    ds_parts = dict(enumerate(numbers))
    # Calculate total sum
    total_sum = sum(ds_parts.values())

    # Calculate desired total for each dataset
    desired_totals = {k: round(columns * rows * v / total_sum) for k, v in ds_parts.items()}
    ok_totals = {k: np.ceil(columns * rows * v / total_sum) for k, v in ds_parts.items()}

    # Spread desired total uniformly across `columns` numbers
    distributed = {}
    for k, total in desired_totals.items():
        avg, rem = divmod(total, columns)
        distributed[k] = [avg] * columns
        for i in range(rem):
            distributed[k][i] += 1

    # Adjusting the numbers iteratively to ensure column-wise sum is `rows`
    def adjust_distribution(distr):
        for i in range(columns):
            # Sort datasets by their current value in this column
            column = [distr[k][i] for k in distr]
            diff = rows - sum(column)
            datasets_sorted = sorted(distr.keys(), key=lambda k: distr[k][i], reverse=(diff > 0))

            while True:
                for k in datasets_sorted:
                    if diff == 0:
                        break
                    if diff > 0 and sum(distr[k]) < ok_totals[k]:
                        distr[k][i] += 1
                        diff -= 1
                    elif diff < 0 and distr[k][i] > 0:
                        distr[k][i] -= 1
                        diff += 1
                if diff == 0:
                    break

    adjust_distribution(distributed)
    result = [distributed[i] for i in range(len(numbers))]
    rng = np.random.default_rng(42)
    result = np.stack(result).T
    rng.shuffle(result, axis=0)
    return list(result.reshape(-1))
