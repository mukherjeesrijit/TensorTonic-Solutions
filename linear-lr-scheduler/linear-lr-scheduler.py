def linear_lr(step, total_steps, initial_lr, final_lr=0.0, warmup_steps=0) -> float:
    """
    Linear warmup (0→initial_lr) then linear decay (initial_lr→final_lr).
    Steps are 0-based; clamp at final_lr after total_steps.
    """

    arr = np.arange(max(step+1,total_steps))

    conditions = [
        (arr < warmup_steps), 
        (arr >= warmup_steps) & (arr <= total_steps),
        (arr > total_steps)
    ]

    choices = [
        arr * initial_lr / warmup_steps,
        final_lr + (initial_lr-final_lr)*(total_steps-arr)/max(total_steps - warmup_steps, 1e-9),
        final_lr
    ]

    result = np.select(conditions, choices, default = 0)

    print(len(arr))
    print(step)

    return result[step]
    