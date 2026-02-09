


You want to replace **invalid rate values**
(i.e. `rate > 100` OR `rate < 0`) with the **mean of valid rates**.

Your current code only checks `> 100` and the mean logic is slightly wrong.

Here is the clean and correct fix:

```python
valid_mean = df.loc[
    (df['task_completion_rate'] >= 0) & (df['task_completion_rate'] <= 100),
    'task_completion_rate'
].mean()

df['task_completion_rate'] = np.where(
    (df['task_completion_rate'] > 100) | (df['task_completion_rate'] < 0),
    valid_mean,
    df['task_completion_rate']
)
```

✅ This does exactly what you wrote in your comment:

> For any rate column where rate > 100 and rate < 0

→ replaced with the mean of valid values (0–100).

This pattern is perfect for your EDA framework’s **cleaning module** as a reusable function.

## Handling duplicated values
```py
duplicated = df.duplicated().sum()

if duplicated > 0:
    df = df.drop_duplicates()
    print(f"{duplicated} duplicated rows removed.")

print(df.shape)
```