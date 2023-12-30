import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import json

with open('cats/general_categories_mapping.json', 'r', encoding='utf-8') as file:
    cat_map = json.load(file)


def process_orders(user_id, orders_data):
    """
    Ф-ция обработки блока orders из json файла
    :param user_id: пользователь
    :param orders_data: orders - последние заказы пользователя - сайт, категория, бренд и другое
    :return: список словарей с признаками orders
    """
    if not orders_data:
        # Возвращаем только user_id, если данных нет
        return {'user_id': user_id}
    count_site = len(orders_data)
    orders_count = 0
    items_count = 0
    items_count_counts = 0
    cats = set()
    brands = set()
    for sites in orders_data:
        orders_count += len(sites['orders'])
        for order in sites['orders']:
            items_count += len(order['items'])
            for item in order['items']:
                items_count_counts += item.get('count', 0) or 0
                categories = item.get('general-category-path', None)
                if type(categories) == list:
                    cats.update(categories)
                elif categories != None:
                    cats.add(categories)
                brand = item.get('brand-id', None)
                if type(brand) == list:
                    brands.update(brand)
                elif brand != None:
                    brands.add(brand)
    mean_orders_per_site = round(orders_count / count_site, 1)
    mean_items_per_order = round(items_count / orders_count, 1)
    try:
        mean_items_count_per_item = round(items_count_counts / items_count, 0)
    except ZeroDivisionError:
        mean_items_count_per_item = 0
    processed_orders = {
        'user_id': user_id,
        'order_site-id_count': count_site,
        'order_orders_mean': mean_orders_per_site,
        'order_items_mean': mean_items_per_order,
        'order_items_count_mean': mean_items_count_per_item,
        'order_cats': cats,
        'order_brands': brands
    }
    return processed_orders


def process_visits(user_id, visits_data):
    """
    Ф-ция обработки блока visits из json файла
    :param user_id: пользователь
    :param visits_data: visits - последние посещения сайтов-партнеров - сайт, длительность, просмотренные товары и прочее
    :return: список словарей с признаками visits
    """
    if not visits_data:
        # Возвращаем только user_id, если данных нет
        return {'user_id': user_id}
    count_site = len(visits_data)
    visits_count = 0
    items_count = 0
    cats = set()
    brands = set()
    duration = 0
    pages = 0
    for sites in visits_data:
        visits_count += len(sites['visits'])
        for visit in sites['visits']:
            items_count += len(visit.get('visited-items', []))
            categories = visit.get('visited-general-categories', None)
            if type(categories) == list:
                cats.update(categories)
            elif categories != None:
                cats.add(categories)
            brand = visit.get('visited-universal-brands', None)
            if type(brand) == list:
                brands.update(brand)
            elif brand != None:
                brands.add(brand)
            duration += visit.get('session-duration', 0)
            pages += visit.get('pages-count', 0)
    mean_visits_per_site = round(visits_count / count_site, 1)
    mean_items_per_visit = round(items_count / visits_count, 1)
    mean_duration_per_visit = round(duration / visits_count, 1)
    mean_pages_per_visit = round(pages / visits_count, 1)
    processed_visits = {
        'user_id': user_id,
        'visit_site-id_count': count_site,
        'visit_visits_mean': mean_visits_per_site,
        'visit_items_mean': mean_items_per_visit,
        'visit_cats': cats,
        'visit_brands': brands,
        'visit_duration_mean': mean_duration_per_visit,
        'visit_pages_mean': mean_pages_per_visit
    }
    return processed_visits


def process_site(user_id, site_data):
    """
    Ф-ция обработки блока site-meta из json файла
    :param user_id: пользователь
    :param site_data: site-meta - агрегат о пользователе по сайту
    :return: список словарей с признаками site-meta
    """
    if not site_data:
        # Возвращаем только user_id, если данных нет
        return {'user_id': user_id}
    count_site = len(site_data)
    recency = 0
    frequency = 0
    monetary = 0
    for sites in site_data:
        recency += sites.get('recency', 0)
        frequency += sites.get('frequency', 0)
        monetary += sites.get('monetary', 0)
    mean_recency_per_site = round(recency / count_site, 1)
    mean_frequency_per_site = round(frequency / count_site, 1)
    mean_monetary_per_site = round(monetary / count_site, 1)
    processed_site = {
        'user_id': user_id,
        'site_site-id_count': count_site,
        'site_recency_mean': mean_recency_per_site,
        'site_frequency_mean': mean_frequency_per_site,
        'site_monetary_mean': mean_monetary_per_site
    }
    return processed_site


def process_exchange(user_id, exchange_data):
    """
    Ф-ция обработки блока exchange-sessions из json файла
    :param user_id: пользователь
    :param exchange_data: exchange-sessions - прошлые посещения витрины подарков
    :return: список словарей с признаками exchange-sessions
    """
    if not exchange_data:
        # Возвращаем только user_id, если данных нет
        return {'user_id': user_id}
    count_sessions = len(exchange_data)
    clicks = 0
    accepted = 0
    for sessions in exchange_data:
        clicks += len(sessions.get('clicks', []))
        if 'accepted-site-id' in sessions:
            accepted += 1
    mean_clicks_per_session = round(clicks / count_sessions, 1)
    mean_accepted_per_session = round(accepted / count_sessions, 1)
    processed_exchange = {
        'user_id': user_id,
        'exchange_sessions_count': count_sessions,
        'exchange_clicks_mean': mean_clicks_per_session,
        'exchange_accepted_mean': mean_accepted_per_session
    }
    return processed_exchange


def process_last(user_id, last_data):
    """
    Ф-ция обработки блока last-visits-in-categories из json файла
    :param user_id: пользователь
    :param last_data: last-visits-in-categories - агрегат по посещениям из разных категорий
    :return: список словарей с признаками last-visits-in-categories
    """
    if not last_data:
        # Возвращаем только user_id, если данных нет
        return {'user_id': user_id}
    cats = set()
    for cat in last_data:
        cats.add(cat.get('category'))
    processed_last = {
        'user_id': user_id,
        'last_cats': cats
    }
    return processed_last


def get_df(data, target=True):
    """
    Функция формирования датафрейма с признаками из файла json
    :param data: данные с файла json
    :param target: по умолчанию True, т.е. таргет включен в датафрейм (для train, val). Для test - target=False
    :return: датафрейм с признаками для дальнейшей предобработки и обучения/валидации/тестирования.
    """
    orders_users = []
    visits_users = []
    site_users = []
    exchange_users = []
    last_cat_users = []
    target_users = []
    for key, value in data.items():
        orders_user = process_orders(key, data[key]['features'].get('orders', None))
        orders_users.append(orders_user)

        visits_user = process_visits(key, data[key]['features'].get('visits', None))
        visits_users.append(visits_user)

        site_user = process_site(key, data[key]['features'].get('site-meta', None))
        site_users.append(site_user)

        exchange_user = process_exchange(key, data[key]['features'].get('exchange-sessions', None))
        exchange_users.append(exchange_user)

        last_cat_user = process_last(key, data[key]['features'].get('last-visits-in-categories', None))
        last_cat_users.append(last_cat_user)

        if target:
            target_user = {'user_id': key, 'target': data[key].get('target', None)}
            target_users.append(target_user)

    orders_df = pd.DataFrame(orders_users)
    visits_df = pd.DataFrame(visits_users)
    site_df = pd.DataFrame(site_users)
    exchange_df = pd.DataFrame(exchange_users)
    last_df = pd.DataFrame(last_cat_users)
    merge_df = orders_df.merge(visits_df, on='user_id', how='outer') \
        .merge(site_df, on='user_id', how='outer') \
        .merge(exchange_df, on='user_id', how='outer') \
        .merge(last_df, on='user_id', how='outer')
    if target:
        target_df = pd.DataFrame(target_users)
        merge_df = merge_df.merge(target_df, on='user_id', how='outer')

    return merge_df


def map_ids_to_categories(id_set):
    """
    Функция преобразования id категории в наименование
    :param id_set: строка с множеством id категорий в датафрейме
    :return: преобразованное множество с наименованиями категорий
    """
    # Пропускаем обработку, если значение NaN или пустое множество
    if not id_set or pd.isna(id_set):
        return id_set
    # Преобразуем каждый ID в соответствующее название категории
    return {cat_map.get(str(cat_id), "Неизвестная категория") for cat_id in id_set}


def replace_in_set(cat_set):
    """
    Функция для замены специфической строки в множестве на 'fashion'
    :param cat_set: строка с множеством категорий
    :return: замененная строка с множеством категорий или та же
    """
    if '"0" => "f", "1" => "a", "2" => "s", "3" => "h", "4" => "i", "5" => "o", "6" => "n"' in cat_set:
        cat_set.remove('"0" => "f", "1" => "a", "2" => "s", "3" => "h", "4" => "i", "5" => "o", "6" => "n"')
        cat_set.add('fashion')
    return cat_set


def preprocessing_df(df, all_categories, unique_last_cats, target=True):
    """
    Функция предобработки данных в датафрейме перед стандартизацией и подачи в модель
    :param df: датафрейм для предобработки
    :param all_categories: уникальные категории товаров из общего словаря категорий
    :param unique_last_cats: категории last товаров из данных train
    :param target: по умолчанию True, т.е. таргет включен в датафрейм (для train, val). Для test - target=False
    :return: готовый датафрейм для стандартизации и подачи в модель
    """
    # Замена NaN на 0 только в столбцах с типом float64
    float_columns = df.select_dtypes(include=['float64']).columns
    df[float_columns] = df[float_columns].fillna(0)

    # Замена NaN на пустые множества в столбцах с категориями и брендами
    col_null_set = ['order_cats', 'order_brands', 'visit_cats', 'visit_brands', 'last_cats']
    for col in col_null_set:
        df[col] = df[col].apply(lambda x: x if x is not np.nan else set())

    df['last_cats'] = df['last_cats'].apply(replace_in_set)

    # Замена id категорий товаров на наименования
    df['order_cats_name'] = df['order_cats'].apply(map_ids_to_categories)
    df['visit_cats_name'] = df['visit_cats'].apply(map_ids_to_categories)

    # Проверяем наличие "Неизвестной категории" в каждом множестве
    df['has_unknown_category'] = df['order_cats_name'].apply(lambda cats: "Неизвестная категория" \
                                                                          in cats if not pd.isna(cats) else False)
    # Подсчет количества случаев с "Неизвестной категорией"
    unknown_category_count = df['has_unknown_category'].sum()
    #print(f"Количество 'Неизвестных категорий' в orders: {unknown_category_count}")
    # Отображаем строки, где есть "Неизвестные категории"
    # unknown_categories_rows = merge_df[merge_df['has_unknown_category']]
    # print(unknown_categories_rows)

    # Проверяем наличие "Неизвестной категории" в каждом множестве
    df['has_unknown_category_1'] = df['visit_cats_name'].apply(lambda cats: "Неизвестная категория" \
                                                                            in cats if not pd.isna(
        cats) else False)
    # Подсчет количества случаев с "Неизвестной категорией"
    unknown_category_count1 = df['has_unknown_category_1'].sum()
    #print(f"Количество 'Неизвестных категорий' в visits: {unknown_category_count1}")

    # Опционально: отображаем строки, где есть "Неизвестные категории"
    # unknown_categories_rows = merge_df[merge_df['has_unknown_category_1']]
    # print(unknown_categories_rows)

    col_standard = ['order_site-id_count', 'order_orders_mean', 'order_items_mean', 'order_items_count_mean', \
                    'visit_site-id_count', 'visit_visits_mean', 'visit_items_mean', 'visit_duration_mean', \
                    'visit_pages_mean', 'site_site-id_count', 'site_recency_mean', 'site_frequency_mean', \
                    'site_monetary_mean', 'exchange_sessions_count', 'exchange_clicks_mean', 'exchange_accepted_mean']

    # Создание нового столбца, который объединяет категории из обоих столбцов
    df['combined_categories'] = df.apply(lambda x: list(set(x['order_cats_name']) | set(x['visit_cats_name'])), axis=1)

    # MultiLabelBinarizer для объединенных категорий
    mlb = MultiLabelBinarizer(classes=all_categories)
    multi_hot = mlb.fit_transform(df['combined_categories'])

    # Создание DataFrame из multi-hot векторов
    multi_hot_df = pd.DataFrame(multi_hot, columns=mlb.classes_)

    # last_cats MultiLabelBinarizer
    mlb_1 = MultiLabelBinarizer(classes=unique_last_cats)
    multi_hot_last = mlb_1.fit_transform(df['last_cats'])
    multi_hot_last_df = pd.DataFrame(multi_hot_last, columns=mlb_1.classes_)
    multi_hot_last_df

    # Объединение с исходным DataFrame
    df_all = pd.concat([df.reset_index(drop=True), multi_hot_df.reset_index(drop=True), \
                        multi_hot_last_df.reset_index(drop=True)], axis=1)

    if target:
        # Словарь для замены пола на числа
        gender_map = {'male': 0, 'female': 1}
        df_all['gender'] = df_all['target'].map(gender_map)

    df_all.drop(['user_id', 'order_cats', 'visit_cats', 'last_cats', 'order_cats_name', 'visit_cats_name', \
                 'has_unknown_category', 'has_unknown_category_1', 'combined_categories', 'order_brands',
                 'visit_brands'], axis=1, inplace=True)
    if target:
        df_all.drop('target', axis=1, inplace=True)

    return df_all
