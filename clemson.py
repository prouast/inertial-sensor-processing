"""Clemson dataset"""

import csv
import os
import datetime as dt
import logging
import glob
import xlrd
import xml.etree.cElementTree as etree
import tensorflow as tf
import numpy as np

FREQUENCY = 15
ACC_SENSITIVITY = 660.0
GYRO_SENSITIVITY = 2.5
DEFAULT_LABEL = "idle"
FLIP_ACC = [-1., 1., 1.]
FLIP_GYRO = [1., -1., -1.]
TIME_FACTOR = 1000000

TRAIN_IDS = ['p005','p006','p007','p013','p015','p016','p020','p021','p022',
  'p025','p026','p027','p030','p031','p033','p036','p037','p038','p043',
  'p044','p045','p048','p050','p051','p054','p055','p056','p059','p060',
  'p061','p065','p066','p067','p070','p071','p072','p077','p078','p079',
  'p082','p083','p084','p087','p088','p089','p092','p093','p095','p099',
  'p100','p101','p104','p105','p106','p109','p110','p111','p115','p116',
  'p117','p120','p121','p122','p129','p130','p131','p136','p137','p138',
  'p142','p143','p144','p148','p150','p151','p157','p158','p159','p162',
  'p164','p165','p170','p171','p172','p175','p176','p177','p180','p181',
  'p182','p186','p187','p188','p192','p194','p195','p201','p202','p204',
  'p207','p208','p209','p218','p219','p220','p226','p229','p230','p233',
  'p234','p235','p241','p242','p244','p247','p248','p251','p256','p257',
  'p259','p263','p264','p265','p268','p269','p270','p273','p274','p275',
  'p278','p279','p280','p283','p284','p285','p291','p292','p293','p309',
  'p311','p312','p320','p322','p324','p331','p332','p334','p338','p341',
  'p343','p353','p361','p368','p384','p392','p396','p406','p410','p411']
VALID_IDS = ['p011','p017','p023','p028','p034','p039','p046','p052',
  'p057','p062','p068','p074','p080','p085','p090','p096','p102','p107',
  'p113','p118','p123','p132','p139','p145','p153','p160','p166','p173',
  'p178','p184','p189','p198','p205','p215','p221','p231','p236','p245',
  'p252','p260','p266','p271','p276','p281','p289','p297','p315','p326',
  'p336','p347','p372','p397','p413']
TEST_IDS = ['p012','p019','p024','p029','p035','p042','p047','p053','p058',
  'p064','p069','p075','p081','p086','p091','p098','p103','p108','p114',
  'p119','p125','p133','p140','p146','p154','p161','p169','p174','p179',
  'p185','p190','p199','p206','p217','p224','p232','p237','p246','p253',
  'p262','p267','p272','p277','p282','p290','p298','p318','p329','p337',
  'p352','p377','p401']
DESSERTS_FOODS = ['ice_cream', 'yogurt_and_ice_cream', 'cupcake',
  'rice_krispie_treat', 'fruit_loop_and_rice_krispie_treat', 'custom_cake',
  'chunky_chocolate_chip_cookie', 'strawberry_shortcake',
  'cereal_lucky_charms_and_cocoa_puffs', 'sweetzza_chocolate_peanut_butter',
  'cherry_cobbler', 'brownie', 'chocolate_cake', 'peach_cobbler',
  'cereal_apple_jacks', 'vanilla_pudding', 'mini_donut', 'frozen_yogurt',
  'custom_desert_pink_cake', 'ice_cream_cone', 'mousse',
  'marshmallow_on_a_stick', 'glazed_donut', 'apple_pie',
  'waffle_and_ice_cream', 'brownie_and_yogurt', 'yellow_cake', 'cheesecake',
  'blueberry_cobbler', 'oatmeal_cookie', 'peanut_butter_chocolate_fudge',
  'm_and_m_cookie', 'caribbean_pie', 'custom_muffin', 'custom_cookie',
  'sugar_cookie', 'banana_pudding', 'pound_cake', 'pudding_cake',
  'double_oatmeal_cookie_with_frosting_filling', 'toffee',
  'snickerdoodle_cookie', 'custom_chocolate_peanut_butter_bars',
  'bran_and_raisin_muffin', 'blueberry_cobbler_and_ice_cream',
  'cinnamon_apples_and_ice_cream', 'waffle_bar', 'chocolate_pudding',
  'sweetzza_cinnamon_pecan', 'sweetzza_apple', 'pancakes', 'spice_cake',
  'cinnamon_bread', 'candied_sweet_potatoes', 'bread_pudding', 'oatmeal',
  'yogurt', 'cereal_rice_krispies', 'cereal_trix_and_honey_nut_cheerios',
  'cereal_reeses_puffs', 'granola_cereal_with_raisins_and_strawberry_yogurt',
  'cereal_trix', 'cereal_life', 'cereal_apple_jacks', 'cereal_lucky_charms',
  'cereal_corn_pops']
DRINKS_FOODS = ['kiwi_juice', 'sprite_zero', 'sweet_tea', 'water', 'apple_juice',
  'whole_milk', 'cranberry_juice', 'coffee', 'kiwi_strawberry_juice', 'sprite',
  'skim_milk', 'unsweet_tea', 'sweet_and_unsweet_tea_mix',
  'mr_pibb_cherry_coke_mellow_yellow_lemonade_powerade_mix', 'diet_coke',
  'coke_zero', 'lemonade_and_sweet_tea_mix', 'powerade', 'chocolate_milk',
  'orange_juice_and_apple_berry', 'coca_cola', 'cherry_coke', 'dr_pepper',
  'lemonade', 'lowfat_milk', 'orange_fanta', 'pink_lemonade', 'vitamin_water',
  'apple_juice_and_water', 'water_and_sweet_tea_mix', 'lowfat_chocolate_milk',
  'grape_juice', 'soy_milk', 'green_tea', 'cranberry_juice_water_mix',
  'root_beer', 'mellow_yellow', 'orange_juice',
  'cranberry_grape_and_sprite_mix', 'mr_pibb', 'chocolate_skim_milk_mix',
  'cranberry_juice_cocktail']
FRUIT_VEG_FOODS = ['custom_fruit_bowl', 'apple', 'banana', 'cantaloupe',
  'pineapple', 'grapefruit', 'orange', 'olives', 'pear_slices', 'melon',
  'blueberries_with_whipped_cream', 'peaches', 'banana_with_peanut_butter',
  'ambrosia_fruit_with_yogurt', 'collard_greens', 'green_beans',
  'steamed_carrots', 'pickle', 'cauliflower', 'peas', 'broccoli',
  'fresh_carrots', 'edamame', 'pears_and_cottage_cheese']
MEAT_DISHES_FOODS = ['hash_sweet_potato_and_bacon', 'roast_beef', 'corn_dog',
  'roast_pork_loin', 'biscuits_and_sausage_gravy', 'spicy_bbq_pork_spare_ribs',
  'roll', 'smokehouse_bbq_station', 'chicken_enchiladas',
  'bbq_turkey_london_broil', 'latin_spiced_pork_roast', 'turkey_meatloaf',
  'sausage_links', 'sausage_strata', 'spice_pork_and_vegetables', 'picadillo',
  'turkey_sliced', 'hush_puppies', 'buffalo_tenders',
  'grilled_italian_sausage_with_onions_and_peppers', 'cajun_roasted_pork_loin',
  'roasted_turkey_breast_with_herbed_gravy', 'hot_dog', 'turkey_bacon',
  'kielbasa', 'fresca_chicken_quesadilla', 'char_sui_braised_pork',
  'grits_and_sausage_links_and_scrambled_eggs', 'sauteed_pollock',
  'fried_shrimp', 'shrimp_masala_and_peas', 'pad_thai_shrimp_station',
  'blackened_tilapia', 'seafood_newburg', 'vegetable_shrimp_sautee',
  'shrimp_masala', 'grilled_ham_steak', 'country_fried_steak',
  'chinese_beef_and_green_pepper_steak', 'grilled_chicken',
  'grilled_jerk_chicken', 'oven_fried_chicken', 'hunan_chicken',
  'baked_honey_bbq_lemon_chicken', 'soy_chicken', 'popcorn_chicken',
  'sweet_and_spicy_chicken_with_asian_vegetables', 'baked_rotisserie_chicken',
  'sweet_and_spicy_chicken', 'hunters_chicken', 'meat_lasagna',
  'southern_frito_pie']
PIZZA_FOODS = ['pepperoni_pizza', 'eggplant_and_broccoli_pizza', 'cheese_pizza',
  'margherita_pizza', 'snickers_pizza', 'pineapple_upside_down_pizza',
  'chicken_bacon_pesto_pizza', 'chicken,_bacon_and_chipotle_ranch_pizza',
  'sausage_pizza', 'bbq_chicken_pizza',
  'chicken,_black_bean,_jalapeno,_and_pico_pizza', 'taco_pizza',
  'mushroom,_caramelized_onion,_and_pepperoni_pizza',
  'strawberry_shortcake_pizza', 'mushroom,_red_pepper_and_pesto_pizza',
  'veggie_pizza', 'mushroom,_red_pepper_and_spinach_pizza',
  'pepperoni_and_sausage_pizza']
RICE_DISHES_FOODS = ['black_beans_and_rice', 'coconut_rice',
  'hunan_chicken_and_rice', 'white_rice', 'mexican_rice', 'cilantro_lime_rice',
  'pork_chop_suey_with_white_rice', 'jasmine_curried_balinese_rice',
  'yellow_rice', 'jasmine_rice', 'brown_rice',
  'shrimp_masala_and_peas_and_rice', 'stir_fry_with_jasmine_rice', 'stir_fry',
  'stir_fry_with_edamame', 'vegetable_stir_fry_with_black_bean_sauce',
  'vegetable_stir_fry_and_rice_noodles']
SALAD_FOODS = ['salad_bar', 'coleslaw_cowboy', 'custom_spinach_salad',
  'caesar_salad', 'mango_salad', 'potato_salad',
  'penne_spinach_and_balsamic_salad', 'coleslaw', 'pasta_salad',
  'wild_rice_and_barley_salad', 'panzanella_crostini_salad', 'rowdys_coleslaw',
  'caesar_salad_station_with_chicken', 'custom_coleslaw', 'vegetable_salad',
  'baked_beans_and_carrots_and_coleslaw', 'caprese']
SANDWICHES_FOODS = ['bbq_brisket_on_kaiser_roll', 'custom_sandwich',
  'custom_whole_wheat_chicken_salad_sandwich', 'homestyle_chicken_sandwich',
  'grilled_cheese_and_tomato_sandwich', 'custom_sandwich_chicken',
  'chicken_sandwich_with_pepper_jack_and_pic', 'chicken_sandwich',
  'grilled_ham_and_cheese_sandwich', 'custom_spinach_wrap_with_chicken_and_ham',
  'israeli_couscous_salad', 'custom_turkey_sub', 'custom_turkey_sandwich',
  'custom_pita_grilled_sandwich', 'pepper_jack_ranch_chicken_wrap',
  'custom_sandwich_italian', 'buffalo_blue_chicken_wrap',
  'custom_sandwich_chicken_salad', 'sloppy_joe',
  'custom_chicken_salad_sandwich',
  'peanut_butter_and_jelly_wrap_with_rice_krispies', 'monte_cristo_sandwich',
  'custom_grilled_chicken_wrap', 'bbq_turkey_sandwich',
  'buffalo_chicken_sandwich', 'grilled_veggie_sub', 'custom_chicken_wrap',
  'chicken_caesar_wrap', 'custom_wrap', 'bbq_pork_sandwich',
  'custom_wrap_spinach', 'california_chicken_wrap',
  'italian_sausage_sandwich_with_peppers_and_onions', 'custom_sandwich_turkey',
  'chicken_sandwich_with_chipotle_mayo', 'custom_sandwich_fried_peanut_butter',
  'reuben_melt', 'vietnamese_pork_sandwich_on_baguette', 'grilled_baguette',
  'grilled_cheese_and_bacon_on_texas_toast', 'hamburger', 'garden_burger',
  'cheeseburger', 'patty_melt', 'vegetable_egg_roll',
  'bacon,_grilled_apple,_and_blue_cheese_bruschetta',
  'soft_shell_pork_carnitas_tacos', 'taco', 'soft_chicken_taco', 'bread',
  'pita_bread', 'ham,_egg,_cheese,_and_salsa_burrito', 'burrito_station']
SNACKS_FOODS = ['signature_chips', 'garlic_breadsticks', 'pretzels',
  'shoestring_french_fries', 'diced_hashbrowns', 'popcorn',
  'french_toast_sticks', 'croutons', 'goldfish']
SOUPS_STEWS_FOODS = ['veggie_gumbo_soup', 'corn_soup', 'seafood_bisque_soup',
  'lintel_soup', 'vegetable_soup', 'squash_soup', 'tomato_basil_soup',
  'chicken_and_corn_soup', 'manhattan_clam_chowder',
  'pho_chicken_broth_bowl_station', 'coconut_curry_soup', 'pork_cadillo_stew',
  'stew_beef', 'polenta_with_broccoli_rabe_and_mushrooms', 'beans_borracho',
  'sweet_creamed_corn']
VEG_DISHES_FOODS = ['overstuffed_potato_station',
  'portobella_mushroom_with_bbq_onions', 'capri_blend_vegetables',
  'eggplant_parmesan', 'ginger_orange_glazed_steamed_carrots', 'refried_beans',
  'baked_potato', 'marinated_tomatoes', 'cauliflower_au_gratin',
  'roasted_garlic_potatoes', 'steamed_spinach_with_lemon_pepper',
  'mashed_potatoes_and_peas', 'asian_vegetables', 'glazed_baby_carrots',
  'steamed_and_seasoned_veggies', 'oven_roasted_red_potatoes',
  'steamed_california_blend_vegetables', 'peas_and_carrots', 'corn_on_the_cob',
  'roasted_sweet_potato', 'seasoned_yellow_squash', 'black_beans_cumin',
  'seasoned_corn', 'african_spiced_sweet_potato',
  'chesapeake_corn_and_tomatoes', 'lyonnaise_potatoes', 'corn_salsa',
  'mashed_potatoes_and_corn', 'fried_plantains', 'baked_beans',
  'steamed_capri_blend_vegetables', 'mashed_red_potatoes', 'chickpeas',
  'russet_mashed_potatoes_and_onions', 'wasabi_mashed_potatoes',
  'succotash_ancho', 'sauteed_tomatoes_and_zucchini', 'calabacitas',
  'broccoli_florets_steamed_with_lemon_zest', 'seasoned_dry_limas',
  'cornbread', 'bread_with_refried_beans', 'pasta_tour_of_italy',
  'macaroni_and_cheese', 'fiesta_pasta',
  'farfalle_pasta_with_broccoli_and_ricotta', 'rotini_with_marinara',
  'veggie_indian_curry', 'tofu_grilled_sesame_seed', 'grilled_bbq_tofu',
  'scrambled_eggs', 'cottage_cheese', 'spinach_and_cheese_quiche', 'falafel',
  'black_bean_cakes', 'eggplant_and_bean_casserole', 'potato_pancakes',
  'hard_boiled_eggs']

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _floats_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


class Dataset():

  def __init__(self, src_dir, exp_dir, dom_hand_spec, label_spec,
    label_spec_inherit, exp_uniform, exp_format):
    self.src_dir = src_dir
    self.exp_dir = exp_dir
    self.dom_hand_spec = dom_hand_spec
    self.label_spec = label_spec
    self.label_spec_inherit = label_spec_inherit
    self.exp_uniform = exp_uniform
    self.exp_format = exp_format
    # Class names
    self.names_1, self.names_2, self.names_3, self.names_4, self.names_5, self.names_6 = \
      self.__class_names()

  def __class_names(self):
    """Get class names from label master file"""
    assert os.path.isfile(self.label_spec), "Couldn't find label_spec file at {}".format(self.label_spec)
    names_1, names_2, names_3, names_4, names_5, names_6 = [], [], [], [], [], []
    tree = etree.parse(self.label_spec)
    categories = tree.getroot()
    for tag in categories[0]:
      names_1.append(tag.attrib['name'])
    for tag in categories[1]:
      names_2.append(tag.attrib['name'])
    for tag in categories[2]:
      names_3.append(tag.attrib['name'])
    for tag in categories[3]:
      names_4.append(tag.attrib['name'])
    for tag in categories[4]:
      names_5.append(tag.attrib['name'])
    for tag in categories[5]:
      names_6.append(tag.attrib['name'])
    return names_1, names_2, names_3, names_4, names_5, names_6

  def __get_food_class(self, food):
    if food in DESSERTS_FOODS:
      return 'dessert'
    elif food in DRINKS_FOODS:
      return 'drink'
    elif food in FRUIT_VEG_FOODS:
      return 'fruit_veg'
    elif food in MEAT_DISHES_FOODS:
      return 'meat_dish'
    elif food in PIZZA_FOODS:
      return 'pizza'
    elif food in RICE_DISHES_FOODS:
      return 'rice_dish'
    elif food in SALAD_FOODS:
      return 'salad'
    elif food in SANDWICHES_FOODS:
      return 'sandwich_wrap'
    elif food in SNACKS_FOODS:
      return 'snack'
    elif food in SOUPS_STEWS_FOODS:
      return 'soup_stew'
    elif food in VEG_DISHES_FOODS:
      return 'veg_dish'
    else:
      return None

  def ids(self):
    data_dir = os.path.join(self.src_dir, "all-data")
    subject_ids = [x for x in next(os.walk(data_dir))[1]]
    ids = []
    for subject_id in subject_ids:
      subject_dir = os.path.join(data_dir, subject_id)
      session_ids = [x for x in next(os.walk(subject_dir))[1]]
      for session_id in session_ids:
        ids.append((subject_id, session_id))
    return ids

  def check(self, id):
    # Path of gesture annotations
    gesture_dir = os.path.join(self.src_dir, "all-gt-gestures", id[0],
      id[1], "gesture_union.txt")
    if not os.path.isfile(gesture_dir):
      logging.warn("No gesture annotations found. Skipping {}_{}.".format(
        id[0], id[1]))
      return False
    # Path of bite annotations
    bite_dir = os.path.join(self.src_dir, "all-gt-bites", id[0],
      id[1], "gt_union.txt")
    if not os.path.isfile(bite_dir):
      logging.warn("No bite annotations found. Skipping {}_{}.".format(
        id[0], id[1]))
      return False
    return True

  def data(self, _, id):
    logging.info("Reading raw data from txt")
    # Read acc and gyro
    dir = os.path.join(self.src_dir, "all-data", id[0], id[1])
    files = glob.glob(os.path.join(dir, "*.txt"))
    assert files, "No raw data found for {} {}".format(id[0], id[1])
    acc = []
    gyro = []
    with open(files[0]) as dest_f:
      # Read voltage values
      v_acc_x = []; v_acc_y = []; v_acc_z = []
      v_gyro_x = []; v_gyro_y = []; v_gyro_z = []
      for row in csv.reader(dest_f, delimiter='\t'):
        v_acc_x.append(float(row[0]))
        v_acc_y.append(float(row[1]))
        v_acc_z.append(float(row[2]))
        v_gyro_x.append(float(row[3]))
        v_gyro_y.append(float(row[4]))
        v_gyro_z.append(float(row[5]))
      # Calculate voltage averages for gyroscope
      v_gyro_x_avg = np.average(v_gyro_x)
      v_gyro_y_avg = np.average(v_gyro_y)
      v_gyro_z_avg = np.average(v_gyro_z)
      # Derive acceleration in g and rotational velocity in deg/s
      for i, vals in enumerate(zip(
        v_acc_x, v_acc_y, v_acc_z, v_gyro_x, v_gyro_y, v_gyro_z)):
        acc_x = (vals[0] - 1.65) * 1000.0 / ACC_SENSITIVITY
        acc_y = (vals[1] - 1.65) * 1000.0 / ACC_SENSITIVITY
        acc_z = (vals[2] - 1.65) * 1000.0 / ACC_SENSITIVITY
        acc.append([acc_x, acc_y, acc_z])
        gyro_x = (vals[3] - v_gyro_x_avg) * 1000.0 / GYRO_SENSITIVITY
        gyro_y = (vals[4] - v_gyro_y_avg) * 1000.0 / GYRO_SENSITIVITY
        gyro_z = (vals[5] - v_gyro_z_avg) * 1000.0 / GYRO_SENSITIVITY
        gyro.append([gyro_x, gyro_y, gyro_z])
    dt = TIME_FACTOR // FREQUENCY # In microseconds
    timestamps = range(0, len(acc)*dt, dt)
    return timestamps, {"hand": (acc, gyro)}

  def dominant(self, id):
    """Read handedness, which is the hand sensor was placed on"""
    file_path = os.path.join(self.src_dir, "demographics.xlsx")
    workbook = xlrd.open_workbook(file_path)
    sheet = workbook.sheet_by_index(0)
    for rowx in range(sheet.nrows):
      cols = sheet.row_values(rowx)
      if cols[0].lower() == id[0]:
        return cols[4].lower()
    return None

  def labels(self, _, id, timestamps):
    def _index_to_ms(index):
      dt = TIME_FACTOR // FREQUENCY
      return index * dt
    # Read gesture ground truth
    gesture_dir = os.path.join(self.src_dir, "all-gt-gestures", id[0],
      id[1], "gesture_union.txt")
    label_1, label_2, start_time, end_time = [], [], [], []
    with open(gesture_dir) as dest_f:
      for row in csv.reader(dest_f, delimiter='\t'):
        if row[0].lower() in self.names_2:
          label_1.append("intake")
          label_2.append(row[0].lower())
          start_time.append(_index_to_ms(int(row[1])))
          end_time.append(_index_to_ms(int(row[2])))
    # Read bite ground truth by matching with gestures
    bite_dir = os.path.join(self.src_dir, "all-gt-bites", id[0],
      id[1], "gt_union.txt")
    num = len(timestamps)
    labels_1 = np.empty(num, dtype='U25'); labels_1.fill(DEFAULT_LABEL)
    labels_2 = np.empty(num, dtype='U25'); labels_2.fill(DEFAULT_LABEL)
    labels_3 = np.empty(num, dtype='U25'); labels_3.fill(DEFAULT_LABEL)
    labels_4 = np.empty(num, dtype='U25'); labels_4.fill(DEFAULT_LABEL)
    labels_5 = np.empty(num, dtype='U25'); labels_5.fill(DEFAULT_LABEL)
    labels_6 = np.empty(num, dtype='U25'); labels_6.fill(DEFAULT_LABEL)
    for l1, l2, start, end in zip(label_1, label_2, start_time, end_time):
      start_frame = np.argmax(np.array(timestamps) >= start)
      end_frame = np.argmax(np.array(timestamps) > end)
      match_found = False
      with open(bite_dir) as dest_f:
        for row in csv.reader(dest_f, delimiter='\t'):
          time = _index_to_ms(int(row[1]))
          if time >= start and time <= end:
            if row[2].lower() in self.names_3:
              l3 = row[2].lower()
            if row[3].lower() in self.names_4:
              l4 = row[3].lower()
            if row[4].lower() in self.names_5:
              l5 = row[4].lower()
            food = self.__get_food_class(row[5].lower())
            if food in self.names_6:
              l6 = food
            else:
              l6 = "NA"
              logging.warn("No food class identified for {}".format(food))
            match_found = True
            break
      if not match_found:
        l3 = "NA"; l4 = "NA"; l5 = "NA"; l6 = "NA"
      labels_1[start_frame:end_frame] = l1
      labels_2[start_frame:end_frame] = l2
      if l3 in self.names_3:
        labels_3[start_frame:end_frame] = l3
      if l4 in self.names_4:
        labels_4[start_frame:end_frame] = l4
      if l5 in self.names_5:
        labels_5[start_frame:end_frame] = l5
      if l6 in self.names_6:
        labels_6[start_frame:end_frame] = l6

    return (labels_1, labels_2, labels_3, labels_4, labels_5, labels_6)

  def write(self, path, id, timestamps, data, dominant_hand, labels):
    frame_ids = range(0, len(timestamps))
    id = '_'.join(id)
    def _format_time(t):
      return (dt.datetime.min + dt.timedelta(microseconds=t)).time().strftime('%H:%M:%S.%f')
    timestamps = [_format_time(t) for t in timestamps]
    acc = np.asarray(data["hand"][0])
    gyro = np.asarray(data["hand"][1])
    assert len(timestamps) == len(acc), \
      "Number timestamps and acc readings must be equal"
    assert len(timestamps) == len(gyro), \
      "Number timestamps and acc readings must be equal"
    if self.exp_format == 'csv':
      with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["id", "frame_id", "timestamp", "acc_x", "acc_y",
          "acc_z", "gyro_x", "gyro_y", "gyro_z", "hand",
          "label_1", "label_2", "label_3", "label_4", "label_5", "label_6"])
        for i in range(0, len(timestamps)):
          writer.writerow([id, frame_ids[i], timestamps[i],
            acc[i][0], acc[i][1], acc[i][2], gyro[i][0], gyro[i][1],
            gyro[i][2], dominant_hand, labels[0][i], labels[1][i],
            labels[2][i], labels[3][i], labels[4][i], labels[5][i]])
    elif self.exp_format == 'tfrecord':
      with tf.io.TFRecordWriter(path) as tfrecord_writer:
        for i in range(0, len(timestamps)):
          example = tf.train.Example(features=tf.train.Features(feature={
            'example/subject_id': _bytes_feature(id.encode()),
            'example/frame_id': _int64_feature(frame_ids[i]),
            'example/timestamp': _bytes_feature(timestamps[i].encode()),
            'example/acc': _floats_feature(acc[i].ravel()),
            'example/gyro': _floats_feature(gyro[i].ravel()),
            'example/label_1': _bytes_feature(labels[0][i].encode()),
            'example/label_2': _bytes_feature(labels[1][i].encode()),
            'example/label_3': _bytes_feature(labels[2][i].encode()),
            'example/label_4': _bytes_feature(labels[3][i].encode()),
            'example/label_5': _bytes_feature(labels[4][i].encode()),
            'example/label_6': _bytes_feature(labels[5][i].encode())
          }))
          tfrecord_writer.write(example.SerializeToString())

  def done(self):
    logging.info("Done")

  def get_flip_signs(self):
    return FLIP_ACC, FLIP_GYRO

  def get_frequency(self):
    return FREQUENCY

  def get_time_factor(self):
    return TIME_FACTOR

  def get_train_ids(self):
    return TRAIN_IDS

  def get_valid_ids(self):
    return VALID_IDS

  def get_test_ids(self):
    return TEST_IDS
