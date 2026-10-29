# Typical mistakes of char-CNN on the real test (D6)

Mistakes: 1390 of 2953 items. The labels were assigned by Claude and only partly re-checked, so a fraction of the 'mistakes' may be label errors.

## Most confident mistakes (one per true class)

| text | true | predicted | p(predicted) |
|---|---|---|---|
| Сосиска в тесте жареная | eating_out.cafe | food.meat_fish | 1.00 |
| Пиво "Халзан безалкогольное" светлое | food.drinks | food.alcohol | 1.00 |
| Лакалют Фикс Набор Крем для фиксации зуб.протезов.шт. | health.pharmacy | clothes_beauty.cosmetics | 1.00 |
| Сырнный латте 400мл на классическом молоке | eating_out.coffee | food.dairy | 1.00 |
| Хот дог с Чоризо 130г 1 шт. | eating_out.fastfood | eating_out.cafe | 0.99 |
| Говядина тушеная "СовПрод" 338 Г(233) | food.grocery | food.meat_fish | 0.99 |
| ПИВО ЗЕДАЗЕНИ 0,5 2 ПОРЦ. | food.alcohol | eating_out.cafe | 0.99 |
| САЛАТ РУККОЛА 75Г | food.produce | eating_out.cafe | 0.99 |
| Изоляция для труб K-Flex EC 42/13 мм. 1 м. | home.repair | clothes_beauty.cosmetics | 0.98 |
| Конверт с курицей и сыром 0,120 | food.bakery | eating_out.cafe | 0.97 |

## Most frequent confusions (true -> predicted)

| true | predicted | count |
|---|---|---|
| eating_out.cafe | food.ready | 351 |
| eating_out.cafe | food.meat_fish | 155 |
| eating_out.cafe | food.grocery | 121 |
| eating_out.cafe | food.drinks | 87 |
| eating_out.cafe | food.produce | 67 |
| eating_out.cafe | food.sweets_snacks | 43 |
| eating_out.cafe | food.dairy | 36 |
| eating_out.cafe | family_kids.goods | 28 |
| eating_out.cafe | eating_out.fastfood | 20 |
| food.bakery | food.grocery | 19 |
