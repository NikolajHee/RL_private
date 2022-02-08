    cs = [s.price_of_order(order) for s in fruit_shops]
    best_shop = fruit_shops[cs.index(min(cs))] 