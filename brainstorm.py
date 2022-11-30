# """
# instabot example
# Workflow:
#     Save users' following into a file.
#
# # simon.fehrer: 4317201099
# # alicia.nmarie: 547319216
# # powergirls_de: 33264303069
# # powerstyle_de: 10607727870
# """
import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path
import yaml
import csv
import sympy

# sys.path.append(os.path.join(sys.path[0], "../"))
from instabot import Bot  # , utils

#
# path_user_lut = Path.cwd() / f'files/user_lut.yaml'
# path_users_done = Path.cwd() / f'files/users_done.yaml'
# path_vornamen_csv = Path.cwd() / f'files/vornamen.csv'
# path_filtered = Path.cwd() / f'files/filtered_ids.yaml'  # todo
#
# # instaUsers = ["simon.fehrer"]
# # 'ich bin verantwortlich für das Ambassador Team von Powergirls. W'
# directMessage = 'Hallo{} ☺️' \
#                 'wir suchen derzeit Models um unsere junge Fitnessmarke bekannter zu machen🤸' \
#                 'Dein Profil gefällt uns sehr gut und wir glauben, dass du perfekt zu uns passen würdest!💪  \n' \
#                 'Hättest du eventuell Lust auf eine Zusammenarbeit mit uns? 🤗 Schau gerne mal auf unserer Website vorbei\n' \
#                 'https://www.powergirls-shop.de\n' \
#                 'Ich könnte für dich am Anfang einen persönlichen Rabattcode von 25-50% einrichten und für deine Follower auch einen für 15 🎀 ' \
#                 'Falls du gerne mehr Informationen hättest, auch zu weiteren Vorteilen/ Provisionen- sag gerne Bescheid 🤗\n' \
#                 'Sportliche Grüße, Alina vom Powergirls Team \n'
#
# write_comments = [#'Sehr schönes Bild ☺ wenn du Lust auf eine Kooperation mit uns hättest, schau doch mal in deine '
#                   'Nachrichten 😊',
#                   'Sehr schönes Bild ☺ schau doch mal in deine '
#                   'Nachrichten 😊🌸',
#                   'Toller Post ☺ wir haben dir mal  '
#                   'geschrieben 🤗',
#                   'Toller Style. Wir suchen derzeit Models- schreib uns gerne mal an ☺️',
#                   'Cool!']
#
# # with Path.open(path_vornamen_csv, 'r') as file:
# #     vornamen = csv.reader(file, delimiter=',')
# #     weiblich = []
# #     maennlich = []
# #     ausschliessen = []
# #     for row in vornamen:
# #         if row[2] == 'w':
# #             weiblich.append(row[0])
# #         elif row[2] == 'm' and len(row[0]) > 4:
# #             maennlich.append(row[0])
# #         elif row[2] == 'a':
# #             ausschliessen.append(row[0])
#
#
# # def assume_gender_weiblich(username):
# #     # for rep in ['_', '.']:
# #     #     username = username.replace(rep, '')
# #
# #     is_weiblich = any([x in username for x in weiblich])
# #     is_maennlich = any([x in username for x in maennlich if len(x) > 4])
# #     is_ausschliessen = any([x in username for x in ausschliessen])
# #
# #     if is_weiblich and is_maennlich:
# #         print(f'User {username} ist weiblich als auch maennlich >4 :P Abort!')
# #         return False
# #     elif is_maennlich:
# #         return False
# #     elif is_ausschliessen:
# #         return False
# #     elif is_weiblich:
# #         return True
# #     else:
# #         return False
#
#
# # def vornamen_anschreiben(username):
# #     # for rep in ['_', '.']:
# #     #     username = username.replace(rep, '')
# #
# #     subset_w = [x for x in weiblich if x in username]
# #     print(f'Erkannte Namen: {subset_w} in {username}')
# #     if len(subset_w) >= 1:
# #         subset_w = sorted(subset_w, key=lambda l: -len(l))
# #         return f' {subset_w[0].capitalize()}'
# #     return ''
#
#
# class AliciasBot:
#
#     def __init__(self):
#         try:
#             with Path.open(path_user_lut, 'r') as file:
#                 self.user_lut = yaml.load(file, Loader=yaml.FullLoader)
#         except Exception as ex:
#             self.user_lut = {}
#
#         try:
#             with Path.open(path_users_done, 'r') as file:
#                 self.users_done = yaml.load(file, Loader=yaml.FullLoader)
#         except Exception as ex:
#             self.users_done = []
#
#         self. self.  # , proxy=args.proxy)
#         self.bot.login(username="powerstyle_de", password="Silvia64")  # , proxy=args.proxy)
#         # self.bot.login(username="powergirls_de", password="Simon63")  # , proxy=args.proxy)
#         # self.bot.login(username="powergirls_de", password="Simon63")  # , proxy=args.proxy)
#         # self.bot.login(username="noobkill@re-gister.com", password="Logiton015@")  # , proxy=args.proxy)
#         # todo kommentar erst bei 2 tage unbeantworteter nachricht und post weiter oben kommentieren
#         # mit bio namen vgl
#         # info = bot.get_user_info('')
#         # print(info)  # todo
#
#     def get_new_potential_ids(self, account_source):
#
#         account_source = account_source or ['model_mgmt']  # todo account_source = account_source or ['go_models_com']
#         for ii, source_acc in enumerate(account_source):
#             path_pot = Path.cwd() / f'files/from_accounts/{source_acc}.yaml'
#
#             try:
#                 with Path.open(path_pot, 'r') as file:
#                     potusers = yaml.load(file, Loader=yaml.FullLoader)
#             except Exception as ex:
#                 potusers = self.bot.get_user_followers(source_acc)
#                 with Path.open(path_pot, 'w') as file:
#                     yaml.dump(potusers, file, default_flow_style=False, sort_keys=False)
#
#         return potusers
#
#     def get_new_potential_ids_from_post(self):
#         source_acc = 'gianninapr'
#         path_pot = Path.cwd() / f'files/from_account_medias/{source_acc}.yaml'
#
#         try:
#             with Path.open(path_pot, 'r') as file:
#                 potusers = yaml.load(file, Loader=yaml.FullLoader)
#         except Exception as ex:
#             medias = self.bot.get_user_medias(source_acc, filtration=False)
#             potuserposts = self.bot.get_media_likers(medias[0])
#             with Path.open(path_pot, 'w') as file:
#                 yaml.dump(potuserposts, file, default_flow_style=False, sort_keys=False)
#
#         return potusers
#
#     # def auto_message_bot(self, max_user_requests=250, max_contacts=20, account_source=None):
#     #     """
#     #     this function does stuff
#     #     maxuserreq: deprecated
#     #
#     #     untere zeile die mit # für post user likes nehmen museclub
#     #     """
#     #     potusers = self.get_new_potential_ids(account_source)
#     #     # potusers = self.get_new_potential_ids_from_post()
#     #     num_contacted = 0
#     #
#     #     for ii, uid in enumerate(potusers):
#     #
#     #         try:
#     #             info = self.user_lut[uid]
#     #         except KeyError as ex:
#     #             info = self.bot.get_user_info(uid)
#     #             self.user_lut[uid] = {k: info[k] for k in
#     #                                   ('pk', 'username', 'full_name', 'is_private', 'profile_pic_url',
#     #                                    'is_verified', 'has_anonymous_profile_picture', 'media_count',
#     #                                    'follower_count', 'following_count', 'biography', 'account_type')}
#     #             # 'whatsapp_number'
#     #         # todo use normal name
#     #         # todo bilder liken: ausnahme
#     #         if uid in self.users_done:
#     #             continue
#     #         if info['is_private']:
#     #             continue
#     #         if info['has_anonymous_profile_picture']:
#     #             continue
#     #         if assume_gender_weiblich(info['username']) and \
#     #                 info['media_count'] >= 1 and \
#     #                 info['follower_count'] > 170:
#     #
#     #             if num_contacted >= max_contacts:
#     #                 raise Exception(f'Passed the maximum of {max_contacts} users!')
#     #             else:
#     #                 num_contacted += 1
#     #             print(f"Kontaktiere Nutzer: {info['username']}")
#     #             self.bot.like_user(uid, random.randint(2, 3), filtration=False)
#     #             time.sleep(random.randint(8, 14))
#     #             contact_message = directMessage.format(vornamen_anschreiben(info['username']))
#     #             time.sleep(random.randint(36, 41))
#     #             self.bot.send_message(contact_message, uid)
#     #
#     #             try:
#     #                 if random.random() > 0.2  :
#     #                     # self.bot.comment_user(uid, amount=1)
#     #                     medias = self.bot.get_user_medias(uid, filtration=False)
#     #                     print(f'Found the following medias: {medias}')
#     #                     self.bot.comment(random.choice(medias[:3]), random.choice(write_comments))
#     #             except Exception as ex:
#     #                 print(f'Omqqq waaaaarum gehts ncht simon hilfe!: {ex}')
#     #             self.users_done.append(uid)
#     #             time.sleep(random.randint(100, 650))  # todo
#     #         else:
#     #             continue
#
#         #
#         # todo: look at last liked posts?
#         # # usernames to get likers from
#         # pages_to_scrape = bot.read_list_from_file("scrape.txt")
#         # f = open("medialikers.txt", "w")  # stored likers in user_ids
#         # for users in pages_to_scrape:
#         #     medias = bot.get_user_medias(users, filtration=False)
#         #     getlikers = bot.get_media_likers(medias[0])
#         #     for likers in getlikers:
#         #         f.write(likers + "\n")
#         # print("succesfully written latest medialikers of" + str(pages_to_scrape))
#         # f.close()
#         #
#
#         # # convert passed user-ids to usernames for usablility
#         # print("Reading from medialikers.txt")
#         # wusers = bot.read_list_from_file("medialikers.txt")
#         # with open("usernames.txt", "w") as f:
#         #     for user_id in wusers:
#         #         username = bot.get_username_from_user_id(user_id)
#         #         f.write(username + "\n")
#         # print("succesfully converted  " + str(wusers))
#         # # parse usernames into a list
#         # with open("usernames.txt", encoding="utf-8") as file:
#         #     instaUsers4 = [l.strip() for l in file]
#         #     bot.send_messages(directMessage, instaUsers4)
#         #     print("Sent An Individual Messages To All Users..")
#
#     def save_users_done(self):
#
#         with Path.open(path_users_done, 'w') as file:
#             yaml.dump(self.users_done, file, default_flow_style=False, sort_keys=False)
#
#     def save_files(self):
#         self.save_users_done()
#         with Path.open(path_user_lut, 'w') as file:
#             yaml.dump(self.user_lut, file, default_flow_style=False, sort_keys=False)
#
#
# if __name__ == '__main__':
#     aliciasBot = AliciasBot()
#     # try:
#     #     aliciasBot.auto_message_bot(max_user_requests=300, max_contacts=25)
#     # except Exception as ex:
#     #     print(f'Ending with exception {ex}')
#     #
#     # aliciasBot.save_files()


bot = Bot(max_likes_per_day=200, follow_delay=random.randint(120, 350), like_delay=random.randint(12, 28),
                       comment_delay=random.randint(18, 34),
                       device='samsung_galaxy_s7')
bot.login(username="schlechtmensch2", password="Logiton015@")

print(bot)
