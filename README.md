# 1)

Install `pyomo` and `pulp`.

**NOTE:** `pyomo` does not have it's own solver (CBC) so we use `pulp`'s solver with `pyomo` as well.

# 2)

Run `main.py`. You should see an output like:

```
Status: 1
Combinations per bar:
bar: 6000
      (2267, 2267, 489, 489) x 1
      (1596, 1596, 489, 489, 489) x 1
      (2267, 489) x 1
```

# 3)

You can find data samples in `test_data.py`. Currently there are only 2, but you can add your own test data as well.
