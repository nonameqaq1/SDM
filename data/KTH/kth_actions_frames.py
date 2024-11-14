# source from https://github.com/buggyyang/RVD/blob/main/data/misc_data_util/kth_actions_frames.py

settings = ['d1', 'd2', 'd3', 'd4']
actions = ['walking', 'jogging', 'running', 'boxing', 'handwaving', 'handclapping']
person_ids = {'train': ['11', '12', '13', '14', '15', '16', '17', '18'],
            'valid': ['19', '20', '21', '23', '24', '25', '01', '04'],
            'test': ['22', '02', '03', '05', '06', '07', '08', '09', '10']}

# MCVD setting
MCVD_person_ids = {
  'train': ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20'],
  'valid': ['21', '22', '23', '24', '25'],
}

# dictionary to store start and end frames for each seqence

kth_actions_dict = {'person01': {'boxing': {'d1': [(1, 95), (96, 185), (186, 245), (246, 360)],
   'd2': [(1, 106), (107, 195), (196, 265), (305, 390)],
   'd3': [(1, 95), (96, 230), (231, 360), (361, 465)],
   'd4': [(1, 106), (107, 170), (171, 245), (246, 370)]},
  'handclapping': {'d1': [(1, 102), (103, 182), (183, 260), (261, 378)],
   'd2': [(1, 162), (163, 282), (358, 472), (473, 550)],
   'd3': [(1, 125), (126, 225), (226, 330), (331, 428)],
   'd4': [(1, 110), (111, 216), (217, 292), (293, 390)]},
  'handwaving': {'d1': [(1, 112), (113, 220), (221, 360), (361, 468)],
   'd2': [(1, 115), (116, 215), (216, 315), (566, 656)],
   'd3': [(1, 112), (113, 252), (253, 356), (357, 456)],
   'd4': [(1, 135), (136, 248), (249, 355), (356, 490)]},
  'jogging': {'d1': [(1, 50), (120, 170), (250, 300), (362, 415)],
   'd2': [(1, 75), (108, 180), (245, 315), (360, 425)],
   'd3': [(1, 50), (115, 164), (254, 300), (360, 405)],
   'd4': [(1, 70), (106, 172), (200, 265), (292, 362)]},
  'running': {'d1': [(1, 35), (95, 130), (202, 235), (295, 335)],
   'd2': [(1, 55), (90, 145), (212, 260), (310, 365)],
   'd3': [(1, 30), (98, 135), (210, 250), (312, 350)],
   'd4': [(1, 45), (65, 110), (142, 190), (486, 530)]},
  'walking': {'d1': [(1, 75), (152, 225), (325, 400), (480, 555)],
   'd2': [(1, 120), (170, 288), (370, 495), (550, 675)],
   'd3': [(1, 90), (172, 250), (352, 430), (875, 950)],
   'd4': [(1, 115), (155, 260), (280, 385), (440, 565)]}},
 'person02': {'boxing': {'d1': [(1, 105), (106, 228), (229, 314), (315, 428)],
   'd2': [(1, 126), (127, 236), (237, 312), (313, 460)],
   'd3': [(1, 120), (121, 218), (219, 350), (351, 490)],
   'd4': [(1, 126), (127, 270), (271, 392), (393, 522)]},
  'handclapping': {'d1': [(1, 102), (103, 216), (217, 308), (309, 398)],
   'd2': [(1, 116), (117, 202), (203, 300), (301, 390)],
   'd3': [(1, 92), (93, 195), (196, 266), (267, 385)],
   'd4': [(1, 100), (101, 235), (236, 312), (313, 386)]},
  'handwaving': {'d1': [(1, 115), (222, 298), (320, 432), (433, 550)],
   'd2': [(1, 82), (83, 186), (187, 260), (356, 440)],
   'd3': [(1, 86), (87, 210), (211, 298), (299, 360)],
   'd4': [(1, 104), (105, 204), (205, 290), (291, 390)]},
  'jogging': {'d1': [(1, 54), (112, 164), (185, 240), (294, 345)],
   'd2': [(1, 152), (168, 290), (380, 492), (512, 626)],
   'd3': [(1, 70), (80, 148), (202, 275), (280, 350)],
   'd4': [(1, 72), (82, 150), (175, 240), (258, 324)]},
  'running': {'d1': [(1, 45), (94, 145), (160, 212), (265, 314)],
   'd2': [(1, 92), (110, 192), (882, 975), (1382, 1492)],
   'd3': [(1, 65), (132, 192), (213, 278), (342, 402)],
   'd4': [(1, 62), (74, 134), (168, 234), (246, 300)]},
  'walking': {'d1': [(1, 72), (150, 228), (230, 295), (372, 450)],
   'd2': [(1, 208), (292, 495), (660, 856), (870, 1060)],
   'd3': [(1, 86), (172, 258), (280, 372), (458, 542)],
   'd4': [(1, 107), (120, 210), (236, 336), (362, 460)]}},
 'person03': {'boxing': {'d1': [(1, 108), (109, 238), (239, 340), (341, 442)],
   'd2': [(1, 128), (129, 225), (226, 352), (353, 502)],
   'd3': [(1, 102), (103, 150), (180, 290), (291, 414)],
   'd4': [(1, 128), (129, 284), (285, 406), (407, 512)]},
  'handclapping': {'d1': [(1, 112), (113, 268), (269, 385), (386, 480)],
   'd2': [(1, 115), (116, 227), (228, 314), (315, 428)],
   'd3': [(1, 142), (143, 228), (229, 345), (346, 460)],
   'd4': [(1, 112), (113, 230), (231, 346), (347, 432)]},
  'handwaving': {'d1': [(1, 134), (135, 266), (267, 400), (401, 570)],
   'd2': [(1, 130), (131, 254), (255, 335), (336, 455)],
   'd3': [(1, 136), (137, 275), (276, 405), (406, 492)],
   'd4': [(1, 128), (214, 340), (341, 465), (466, 585)]},
  'jogging': {'d1': [(1, 58), (122, 175), (300, 355), (435, 492)],
   'd2': [(1, 122), (165, 300), (354, 465), (495, 620)],
   'd3': [(1, 48), (98, 142), (236, 282), (334, 380)],
   'd4': [(1, 64), (114, 174), (192, 252), (300, 360)]},
  'running': {'d1': [(1, 38), (104, 142), (222, 262), (336, 380)],
   'd2': [(1, 90), (102, 188), (850, 925), (955, 1032)],
   'd3': [(1, 34), (92, 126), (232, 268), (342, 375)],
   'd4': [(1, 45), (78, 116), (135, 180), (214, 265)]},
  'walking': {'d1': [(1, 104), (228, 330), (425, 526), (550, 640)],
   'd2': [(1, 226), (238, 468), (478, 685), (765, 960)],
   'd3': [(1, 128), (754, 834), (890, 975), (1032, 1120)],
   'd4': [(1, 92), (118, 216), (232, 338), (339, 432)]}},
 'person04': {'boxing': {'d1': [(1, 106), (107, 184), (185, 282), (283, 328)],
   'd2': [(1, 116), (117, 205), (206, 344), (345, 475)],
   'd3': [(1, 112), (113, 216), (217, 280), (281, 390)],
   'd4': [(1, 88), (89, 220), (221, 312), (313, 378)]},
  'handclapping': {'d1': [(1, 106), (107, 235), (236, 315), (316, 393)],
   'd2': [(1, 85), (86, 195), (196, 328), (329, 460)],
   'd3': [(1, 100), (101, 200), (201, 280), (281, 382)],
   'd4': [(1, 112), (113, 224), (225, 360), (361, 418)]},
  'handwaving': {'d1': [(1, 110), (111, 255), (380, 534), (535, 732)],
   'd2': [(1, 124), (125, 252), (253, 422), (423, 562)],
   'd3': [(1, 112), (113, 264), (265, 380), (381, 538)],
   'd4': [(1, 132), (133, 258), (259, 380), (381, 460)]},
  'jogging': {'d1': [(1, 50), (86, 142), (201, 254), (278, 335)],
   'd2': [(1, 85), (148, 250), (270, 365), (400, 510)],
   'd3': [(1, 50), (88, 140), (185, 238), (275, 325)],
   'd4': [(1, 70), (82, 145), (152, 225), (235, 302)]},
  'running': {'d1': [(1, 35), (58, 98), (142, 180), (202, 242)],
   'd2': [(1, 78), (115, 172), (210, 270), (302, 370)],
   'd3': [(1, 35), (70, 105), (145, 180), (218, 255)],
   'd4': [(1, 50), (62, 108), (120, 170), (172, 230)]},
  'walking': {'d1': [(1, 85), (108, 196), (268, 356), (382, 470)],
   'd2': [(1, 155), (212, 380), (485, 630), (680, 840)],
   'd3': [(1, 74), (128, 215), (272, 350), (394, 480)],
   'd4': [(1, 100), (112, 205), (218, 310), (328, 418)]}},
 'person05': {'boxing': {'d1': [(1, 136), (137, 237), (238, 372), (373, 438)],
   'd2': [(1, 78), (79, 215), (216, 326), (327, 444)],
   'd3': [(1, 96), (97, 225), (226, 352), (353, 430)],
   'd4': [(1, 126), (127, 232), (233, 338), (339, 450)]},
  'handclapping': {'d1': [(1, 128), (129, 220), (221, 330), (331, 422)],
   'd2': [(1, 102), (103, 234), (235, 350), (351, 428)],
   'd3': [(1, 124), (125, 224), (225, 310), (311, 435)],
   'd4': [(1, 100), (101, 198), (199, 328), (329, 455)]},
  'handwaving': {'d1': [(1, 140), (141, 246), (247, 352), (353, 460)],
   'd2': [(1, 136), (137, 232), (233, 366), (367, 468)],
   'd3': [(1, 128), (129, 224), (225, 322), (323, 452)],
   'd4': [(1, 138), (139, 236), (237, 332), (333, 432)]},
  'jogging': {'d1': [(1, 45), (110, 156), (212, 260), (330, 378)],
   'd2': [(1, 62), (128, 190), (226, 288), (342, 400)],
   'd3': [(1, 54), (95, 148), (188, 235), (304, 352)],
   'd4': [(1, 64), (106, 170), (180, 240), (300, 365)]},
  'running': {'d1': [(1, 36), (80, 118), (170, 202), (252, 285)],
   'd2': [(1, 45), (98, 140), (188, 232), (290, 334)],
   'd3': [(1, 40), (86, 126), (185, 220), (282, 322)],
   'd4': [(1, 52), (98, 145), (155, 202), (260, 312)]},
  'walking': {'d1': [(1, 75), (140, 215), (312, 382), (452, 535)],
   'd2': [(1, 115), (208, 318), (365, 480), (560, 675)],
   'd3': [(1, 85), (172, 258), (318, 395), (475, 552)],
   'd4': [(1, 102), (142, 240), (252, 348), (408, 515)]}},
 'person06': {'boxing': {'d1': [(1, 102), (103, 220), (221, 302), (303, 412)],
   'd2': [(1, 86), (87, 160), (161, 296), (297, 408)],
   'd3': [(1, 112), (113, 206), (207, 304), (305, 392)],
   'd4': [(1, 96), (97, 195), (196, 288), (289, 392)]},
  'handclapping': {'d1': [(1, 108), (109, 222), (223, 308), (309, 416)],
   'd2': [(1, 95), (96, 208), (276, 345), (355, 462)],
   'd3': [(1, 100), (101, 190), (191, 260), (261, 378)],
   'd4': [(1, 88), (89, 196), (197, 282), (283, 372)]},
  'handwaving': {'d1': [(1, 110), (111, 264), (265, 382), (383, 460)],
   'd2': [(1, 140), (141, 238), (239, 372), (373, 478)],
   'd3': [(1, 108), (109, 214), (215, 358), (359, 432)],
   'd4': [(1, 112), (113, 256), (257, 359), (360, 430)]},
  'jogging': {'d1': [(1, 50), (78, 128), (186, 242), (243, 305)],
   'd2': [(1, 80), (112, 194), (292, 378), (412, 498)],
   'd3': [(1, 52), (88, 140), (176, 230), (250, 304)],
   'd4': [(1, 75), (102, 168), (176, 238), (272, 340)]},
  'running': {'d1': [(1, 35), (75, 115), (164, 205), (242, 285)],
   'd2': [(1, 52), (84, 150), (222, 286), (318, 385)],
   'd3': [(1, 35), (69, 105), (150, 188), (224, 266)],
   'd4': [(1, 48), (72, 118), (310, 360), (392, 442)]},
  'walking': {'d1': [(1, 72), (126, 192), (240, 308), (350, 420)],
   'd2': [(1, 180), (243, 390), (528, 668), (732, 868)],
   'd3': [(1, 68), (102, 166), (238, 306), (340, 408)],
   'd4': [(1, 100), (132, 230), (240, 332), (382, 478)]}},
 'person07': {'boxing': {'d1': [(1, 108), (109, 208), (209, 290), (291, 385)],
   'd2': [(1, 96), (109, 218), (265, 325), (426, 478)],
   'd3': [(1, 104), (105, 198), (199, 283), (284, 384)],
   'd4': [(1, 102), (103, 191), (192, 268), (269, 368)]},
  'handclapping': {'d1': [(1, 128), (129, 220), (221, 346), (347, 446)],
   'd2': [(1, 102), (103, 248), (249, 368), (369, 518)],
   'd3': [(1, 132), (133, 233), (234, 332), (333, 500)],
   'd4': [(1, 158), (159, 268), (269, 405), (406, 550)]},
  'handwaving': {'d1': [(1, 226), (227, 375), (376, 614), (615, 782)],
   'd2': [(1, 204), (205, 422), (423, 590), (591, 752)],
   'd3': [(1, 150), (151, 316), (317, 428), (429, 595)],
   'd4': [(1, 216), (217, 404), (405, 585), (586, 780)]},
  'jogging': {'d1': [(1, 42), (84, 126), (206, 248), (282, 330)],
   'd2': [(1, 72), (150, 222), (234, 308), (382, 456)],
   'd3': [(1, 50), (80, 132), (186, 240), (262, 312)],
   'd4': [(1, 64), (270, 332), (194, 264), (490, 570)]},
  'running': {'d1': [(1, 28), (78, 110), (192, 220), (280, 312)],
   'd2': [(1, 56), (158, 204), (262, 310), (492, 544)],
   'd3': [(1, 40), (74, 116), (236, 275), (304, 345)],
   'd4': [(1, 52), (68, 115), (310, 358), (224, 274)]},
  'walking': {'d1': [(1, 64), (148, 215), (288, 350), (412, 475)],
   'd2': [(1, 98), (186, 284), (342, 448), (528, 636)],
   'd3': [(1, 88), (112, 194), (298, 386), (414, 495)],
   'd4': [(1, 110), (164, 250), (296, 386), (442, 550)]}},
 'person08': {'boxing': {'d1': [(1, 110), (111, 248), (249, 334), (335, 420)],
   'd2': [(1, 112), (113, 227), (228, 316), (317, 464)],
   'd3': [(1, 68), (108, 216), (217, 354), (355, 434)],
   'd4': [(1, 124), (125, 218), (219, 346), (347, 470)]},
  'handclapping': {'d1': [(1, 108), (109, 190), (191, 296), (349, 458)],
   'd2': [(1, 108), (109, 181), (182, 308), (309, 386)],
   'd3': [(1, 100), (101, 162), (210, 314), (315, 442)],
   'd4': [(1, 100), (101, 232), (233, 340), (341, 416)]},
  'handwaving': {'d1': [(1, 118), (119, 242), (243, 322), (323, 502)],
   'd2': [(1, 108), (109, 216), (217, 358), (359, 430)],
   'd3': [(1, 116), (117, 240), (340, 500), (501, 575)],
   'd4': [(1, 116), (117, 230), (231, 398), (399, 478)]},
  'jogging': {'d1': [(1, 58), (128, 182), (216, 270), (344, 400)],
   'd2': [(1, 84), (132, 212), (258, 345), (398, 478)],
   'd3': [(1, 55), (122, 182), (212, 272), (336, 400)],
   'd4': [(1, 78), (132, 210), (220, 295), (352, 425)]},
  'running': {'d1': [(1, 42), (92, 130), (168, 208), (268, 306)],
   'd2': [(1, 56), (100, 156), (205, 258), (304, 365)],
   'd3': [(1, 42), (104, 150), (198, 248), (638, 686)],
   'd4': [(1, 52), (78, 128), (142, 190), (224, 274)]},
  'walking': {'d1': [(1, 78), (185, 258), (326, 398), (478, 552)],
   'd2': [(1, 128), (200, 330), (408, 542), (622, 750)],
   'd3': [(1, 85), (180, 255), (340, 428), (502, 585)],
   'd4': [(1, 132), (190, 308), (338, 448), (500, 625)]}},
 'person09': {'boxing': {'d1': [(1, 120), (121, 218), (219, 312), (313, 448)],
   'd2': [(1, 128), (129, 205), (206, 330), (331, 432)],
   'd3': [(1, 118), (119, 190), (191, 288), (289, 418)],
   'd4': [(1, 118), (119, 218), (219, 290), (291, 415)]},
  'handclapping': {'d1': [(1, 92), (93, 188), (189, 308), (309, 382)],
   'd2': [(1, 90), (91, 161), (162, 232), (233, 358)],
   'd3': [(1, 85), (86, 195), (196, 264), (265, 356)],
   'd4': [(1, 90), (91, 178), (179, 296), (297, 390)]},
  'handwaving': {'d1': [(1, 108), (109, 222), (223, 372), (373, 450)],
   'd2': [(1, 108), (109, 238), (239, 340), (341, 446)],
   'd3': [(1, 92), (93, 188), (189, 312), (313, 408)],
   'd4': [(1, 97), (98, 202), (203, 270), (271, 405)]},
  'jogging': {'d1': [(1, 50), (144, 192), (272, 315), (412, 465)],
   'd2': [(1, 85), (176, 256), (294, 368), (460, 540)],
   'd3': [(1, 50), (136, 180), (266, 312), (396, 440)],
   'd4': [(1, 80), (110, 184), (186, 248), (286, 350)]},
  'running': {'d1': [(1, 36), (100, 134), (190, 225), (290, 325)],
   'd2': [(1, 54), (112, 170), (192, 250), (316, 370)],
   'd3': [(1, 36), (98, 130), (194, 230), (290, 325)],
   'd4': [(1, 55), (75, 125), (128, 180), (205, 256)]},
  'walking': {'d1': [(1, 65), (208, 275), (390, 460), (618, 685)],
   'd2': [(1, 106), (250, 368), (425, 544), (686, 805)],
   'd3': [(1, 64), (195, 256), (375, 442), (576, 640)],
   'd4': [(1, 90), (118, 208), (218, 302), (334, 422)]}},
 'person10': {'boxing': {'d1': [(1, 92), (93, 188), (189, 286), (287, 422)],
   'd2': [(1, 140), (195, 287), (288, 398), (429, 530)],
   'd3': [(1, 120), (121, 244), (245, 395), (396, 550)],
   'd4': [(1, 130), (131, 233), (234, 368), (369, 474)]},
  'handclapping': {'d1': [(1, 108), (109, 200), (201, 260), (261, 284)],
   'd2': [(1, 64), (65, 162), (163, 255), (256, 359)],
   'd3': [(1, 108), (109, 202), (203, 308), (309, 395)],
   'd4': [(1, 92), (93, 182), (183, 257), (258, 336)]},
  'handwaving': {'d1': [(1, 122), (123, 242), (243, 402), (403, 524)],
   'd2': [(1, 98), (99, 216), (217, 370), (371, 525)],
   'd3': [(1, 110), (111, 265), (266, 395), (396, 518)],
   'd4': [(1, 142), (143, 255), (256, 375), (376, 532)]},
  'jogging': {'d1': [(1, 50), (135, 190), (230, 290), (365, 425)],
   'd2': [(1, 112), (175, 285), (315, 435), (530, 655)],
   'd3': [(1, 50), (136, 192), (215, 256), (350, 400)],
   'd4': [(1, 60), (110, 170), (171, 235), (295, 363)]},
  'running': {'d1': [(1, 35), (95, 125), (180, 210), (290, 322)],
   'd2': [(1, 65), (140, 215), (262, 340), (410, 480)],
   'd3': [(1, 40), (118, 155), (195, 230), (330, 365)],
   'd4': [(1, 50), (100, 145), (152, 195), (246, 290)]},
  'walking': {'d1': [(1, 85), (215, 302), (375, 455), (572, 655)],
   'd2': [(1, 190), (335, 524), (585, 762), (900, 1092)],
   'd3': [(1, 80), (215, 295), (330, 405), (535, 615)],
   'd4': [(1, 95), (162, 270), (271, 375), (435, 544)]}},
 'person11': {'boxing': {'d1': [(1, 74), (75, 141), (142, 273), (274, 351)],
   'd2': [(1, 92), (165, 216), (228, 306), (390, 545)],
   'd3': [(1, 95), (96, 208), (209, 334), (335, 444)],
   'd4': [(1, 106), (107, 194), (195, 250)]},
  'handclapping': {'d1': [(1, 80), (81, 168), (169, 225), (226, 320)],
   'd2': [(1, 95), (96, 133), (134, 225), (226, 300)],
   'd3': [(1, 105), (106, 190), (191, 255), (256, 374)],
   'd4': [(1, 85), (86, 155), (268, 334), (345, 401)]},
  'handwaving': {'d1': [(1, 148), (149, 268), (269, 350), (351, 475)],
   'd2': [(1, 85), (86, 155), (209, 285), (347, 429)],
   'd3': [(1, 135), (136, 232), (233, 390), (391, 515)],
   'd4': [(1, 122), (123, 223), (224, 358), (359, 456)]},
  'jogging': {'d1': [(1, 60), (115, 175), (232, 288), (335, 400)],
   'd2': [(1, 70), (125, 200), (256, 325), (378, 450)],
   'd3': [(1, 60), (130, 178), (250, 305), (365, 415)],
   'd4': [(1, 60), (98, 158), (215, 285), (326, 390)]},
  'running': {'d1': [(1, 35), (140, 180), (274, 310), (415, 450)],
   'd2': [(1, 45), (140, 185), (295, 340), (430, 475)],
   'd3': [(1, 35), (140, 170), (288, 320), (430, 460)],
   'd4': [(1, 45), (110, 155), (188, 245), (280, 320)]},
  'walking': {'d1': [(1, 95), (145, 245), (370, 460), (525, 640)],
   'd2': [(1, 185), (260, 435), (450, 605), (660, 810)],
   'd3': [(1, 90), (180, 310), (385, 475), (590, 680)],
   'd4': [(1, 95), (140, 235), (300, 400), (432, 520)]}},
 'person12': {'boxing': {'d1': [(1, 137), (138, 242), (243, 380), (381, 590)],
   'd2': [(1, 89), (90, 152), (180, 325), (326, 687)],
   'd3': [(1, 155), (156, 300), (310, 560), (561, 701)],
   'd4': [(1, 100), (136, 255), (256, 408), (409, 519)]},
  'handclapping': {'d1': [(1, 125), (126, 220), (221, 316), (317, 392)],
   'd2': [(1, 150), (151, 268), (269, 360), (361, 460)],
   'd3': [(1, 123), (124, 215), (216, 310), (311, 471)],
   'd4': [(1, 133), (134, 237), (238, 344), (345, 479)]},
  'handwaving': {'d1': [(1, 168), (169, 285), (286, 518), (519, 672)],
   'd2': [(1, 112), (182, 325), (326, 438), (439, 590)],
   'd3': [(1, 178), (179, 352), (353, 530), (531, 699)],
   'd4': [(1, 152), (153, 332), (333, 452), (453, 567)]},
  'jogging': {'d1': [(1, 80), (125, 175), (500, 550), (615, 660)],
   'd2': [(1, 80), (140, 225), (285, 365), (425, 510)],
   'd3': [(1, 60), (130, 180), (230, 285), (355, 410)],
   'd4': [(1, 80), (112, 180), (181, 260), (292, 359)]},
  'running': {'d1': [(1, 30), (75, 105), (160, 190), (245, 280)],
   'd2': [(1, 50), (100, 150), (215, 260), (320, 380)],
   'd3': [(1, 40), (100, 135), (188, 225), (290, 330)],
   'd4': [(1, 50), (90, 135), (155, 205), (250, 294)]},
  'walking': {'d1': [(1, 80), (150, 250), (320, 405), (485, 570)],
   'd2': [(1, 160), (225, 400), (472, 620), (690, 850)],
   'd3': [(1, 75), (210, 290), (300, 385), (430, 510)],
   'd4': [(1, 125), (170, 300), (320, 440), (490, 640)]}},
 'person13': {'boxing': {'d1': [(1, 105), (106, 217), (218, 330), (331, 538)],
   'd2': [(1, 144), (171, 295), (296, 415), (416, 631)],
   'd3': [(1, 80), (81, 212), (213, 296), (297, 419)],
   'd4': [(1, 109), (110, 239), (240, 352), (353, 420)]},
  'handclapping': {'d1': [(1, 124), (125, 223), (224, 322), (323, 396)],
   'd2': [(1, 100), (101, 275), (305, 400), (401, 523)],
   'd4': [(1, 132), (133, 234), (235, 335), (336, 443)]},
  'handwaving': {'d1': [(1, 80), (100, 255), (256, 407), (408, 520)],
   'd2': [(1, 116), (117, 225), (226, 333), (334, 492)],
   'd3': [(1, 125), (126, 240), (241, 336)],
   'd4': [(1, 180), (181, 288), (289, 383), (384, 556)]},
  'jogging': {'d1': [(1, 50), (120, 175), (235, 285), (425, 480)],
   'd2': [(1, 65), (175, 245), (325, 400), (505, 585)],
   'd3': [(1, 55), (135, 190), (265, 320), (425, 480)],
   'd4': [(1, 75), (120, 190), (235, 310), (355, 435)]},
  'running': {'d1': [(1, 45), (165, 210), (255, 300), (415, 465)],
   'd2': [(1, 55), (150, 210), (265, 325), (420, 485)],
   'd3': [(1, 50), (140, 185), (255, 300), (400, 450)],
   'd4': [(1, 65), (110, 170), (192, 260), (295, 351)]},
  'walking': {'d1': [(1, 90), (185, 290), (335, 430), (585, 690)],
   'd2': [(1, 150), (345, 485), (520, 650), (800, 940)],
   'd3': [(1, 100), (205, 310), (370, 470), (580, 670)],
   'd4': [(1, 120), (185, 295), (335, 440), (495, 600)]}},
 'person14': {'boxing': {'d1': [(1, 123), (124, 258), (259, 444), (445, 558)],
   'd2': [(1, 102), (155, 264), (265, 415), (416, 610)],
   'd3': [(1, 162), (163, 322), (323, 423), (424, 562)],
   'd4': [(1, 92), (93, 205), (206, 312), (313, 435)]},
  'handclapping': {'d1': [(1, 112), (113, 230), (231, 345), (346, 524)],
   'd2': [(1, 150), (151, 266), (267, 380), (381, 530)],
   'd3': [(1, 110), (111, 262), (263, 375), (376, 515)],
   'd4': [(1, 118), (119, 218), (219, 348), (349, 464)]},
  'handwaving': {'d1': [(1, 165), (166, 305), (306, 445), (446, 545)],
   'd2': [(1, 125), (126, 292), (293, 395), (432, 526)],
   'd3': [(1, 145), (146, 235), (236, 330)],
   'd4': [(1, 132), (133, 265), (266, 446), (447, 631)]},
  'jogging': {'d1': [(1, 55), (165, 245), (284, 345), (460, 525)],
   'd2': [(1, 80), (165, 255), (302, 380), (470, 560)],
   'd3': [(1, 75), (166, 245), (350, 420), (540, 612)],
   'd4': [(1, 70), (95, 165), (217, 300), (325, 415)]},
  'running': {'d1': [(1, 55), (150, 220), (260, 310), (410, 449)],
   'd2': [(1, 65), (155, 210), (265, 320), (425, 477)],
   'd3': [(1, 33), (155, 200), (300, 350), (500, 550)],
   'd4': [(1, 50), (105, 165), (215, 270), (325, 390)]},
  'walking': {'d1': [(1, 115), (205, 350), (395, 500), (590, 720)],
   'd2': [(1, 150), (240, 385), (462, 610), (720, 870)],
   'd3': [(1, 95), (255, 380), (420, 515), (620, 713)],
   'd4': [(1, 115), (155, 280), (290, 400), (440, 555)]}},
 'person15': {'boxing': {'d1': [(1, 150), (151, 216), (217, 290), (291, 408)],
   'd2': [(1, 128), (129, 252), (253, 335), (336, 485)],
   'd3': [(1, 90), (91, 191), (192, 264), (265, 418)],
   'd4': [(1, 90), (91, 200), (201, 294)]},
  'handclapping': {'d1': [(1, 65), (66, 164), (165, 232), (233, 312)],
   'd2': [(1, 85), (86, 155), (156, 228), (229, 285)],
   'd3': [(1, 60), (61, 160), (161, 255), (256, 365)],
   'd4': [(1, 80), (81, 150), (151, 220), (221, 322)]},
  'handwaving': {'d1': [(1, 136), (137, 279), (280, 440), (441, 581)],
   'd2': [(1, 170), (171, 331), (332, 548), (549, 703)],
   'd3': [(1, 123), (124, 262), (263, 404), (405, 598)],
   'd4': [(1, 150), (151, 312), (313, 496), (497, 666)]},
  'jogging': {'d1': [(1, 75), (140, 205), (240, 300), (360, 420)],
   'd2': [(1, 75), (165, 242), (282, 355), (455, 535)],
   'd3': [(1, 50), (130, 180), (242, 300), (385, 440)],
   'd4': [(1, 85), (112, 200), (210, 290), (312, 400)]},
  'running': {'d1': [(1, 50), (100, 150), (180, 225), (280, 325)],
   'd2': [(1, 55), (135, 190), (235, 290), (385, 435)],
   'd3': [(1, 45), (100, 140), (200, 245), (300, 344)],
   'd4': [(1, 55), (92, 150), (170, 225), (265, 320)]},
  'walking': {'d1': [(1, 105), (215, 338), (422, 540), (630, 741)],
   'd2': [(1, 175), (235, 375), (435, 575), (650, 790)],
   'd3': [(1, 80), (175, 258), (365, 455), (565, 655)],
   'd4': [(1, 145), (170, 322), (340, 470), (535, 678)]}},
 'person16': {'boxing': {'d1': [(1, 105), (106, 225), (226, 362), (363, 494)],
   'd2': [(1, 42), (74, 200), (201, 280), (342, 466)],
   'd3': [(1, 120), (121, 250), (251, 370), (371, 530)],
   'd4': [(1, 120), (121, 208), (209, 365), (366, 460)]},
  'handclapping': {'d1': [(1, 100), (101, 235), (236, 338), (339, 470)],
   'd2': [(1, 115), (116, 202), (203, 304), (334, 448)],
   'd3': [(1, 110), (111, 210), (211, 340), (341, 485)],
   'd4': [(1, 104), (105, 192), (193, 278), (279, 390)]},
  'handwaving': {'d1': [(1, 105), (106, 244), (245, 349), (350, 492)],
   'd2': [(1, 108), (109, 200), (246, 352), (354, 497)],
   'd3': [(1, 112), (113, 258), (259, 365), (366, 508)],
   'd4': [(1, 104), (105, 206), (207, 311), (312, 456)]},
  'jogging': {'d1': [(1, 60), (120, 190), (240, 300), (360, 428)],
   'd2': [(1, 90), (135, 225), (285, 365), (415, 510)],
   'd3': [(1, 60), (150, 200), (282, 340), (430, 490)],
   'd4': [(1, 75), (110, 190), (220, 300), (325, 405)]},
  'running': {'d1': [(1, 50), (105, 152), (218, 270), (320, 380)],
   'd2': [(1, 65), (125, 190), (245, 315), (375, 445)],
   'd3': [(1, 50), (130, 185), (256, 300), (390, 440)],
   'd4': [(1, 55), (105, 160), (188, 235), (275, 325)]},
  'walking': {'d1': [(1, 95), (158, 252), (335, 445), (500, 605)],
   'd2': [(1, 160), (225, 360), (420, 560), (610, 750)],
   'd3': [(1, 95), (230, 330), (370, 485), (585, 695)],
   'd4': [(1, 115), (155, 280), (306, 415), (455, 565)]}},
 'person17': {'boxing': {'d1': [(1, 110), (111, 192), (193, 306), (307, 378)],
   'd2': [(1, 112), (113, 198), (199, 335), (336, 490)],
   'd3': [(1, 122), (123, 226), (227, 332), (333, 472)],
   'd4': [(1, 92), (93, 156), (157, 254), (255, 342)]},
  'handclapping': {'d1': [(1, 90), (91, 200), (201, 280), (281, 405)],
   'd2': [(1, 100), (132, 200), (256, 382), (424, 520)],
   'd3': [(1, 90), (91, 208), (209, 325), (326, 450)],
   'd4': [(1, 92), (93, 168), (169, 264), (265, 385)]},
  'handwaving': {'d1': [(1, 86), (87, 198), (199, 286), (287, 430)],
   'd2': [(1, 102), (312, 414), (415, 512), (755, 824)],
   'd3': [(1, 102), (103, 235), (304, 410), (411, 582)],
   'd4': [(1, 108), (109, 190), (191, 292), (293, 394)]},
  'jogging': {'d1': [(1, 60), (160, 220), (285, 345), (440, 500)],
   'd2': [(1, 105), (115, 220), (315, 410), (460, 560)],
   'd3': [(1, 50), (130, 185), (285, 350), (430, 485)],
   'd4': [(1, 70), (90, 160), (175, 240), (255, 320)]},
  'running': {'d1': [(1, 30), (95, 130), (200, 240), (310, 345)],
   'd2': [(1, 60), (115, 165), (260, 312), (370, 430)],
   'd3': [(1, 35), (115, 150), (235, 270), (360, 400)],
   'd4': [(1, 55), (70, 130), (140, 195), (215, 270)]},
  'walking': {'d1': [(1, 115), (160, 270), (350, 460), (570, 666)],
   'd2': [(1, 160), (340, 496), (588, 750), (898, 1065)],
   'd3': [(1, 105), (202, 305), (455, 545), (640, 725)],
   'd4': [(1, 112), (140, 255), (300, 420), (445, 560)]}},
 'person18': {'boxing': {'d1': [(1, 132), (133, 262), (263, 426), (427, 615)],
   'd2': [(1, 85), (115, 211), (271, 376), (412, 540)],
   'd3': [(1, 128), (129, 290), (291, 398), (399, 478)],
   'd4': [(1, 112), (113, 256), (257, 365), (366, 490)]},
  'handclapping': {'d1': [(1, 125), (126, 288), (289, 400), (401, 510)],
   'd2': [(1, 108), (109, 220), (221, 320), (375, 500)],
   'd3': [(1, 128), (129, 300), (301, 425), (426, 550)],
   'd4': [(1, 120), (121, 255), (256, 345), (346, 535)]},
  'handwaving': {'d1': [(1, 136), (137, 280), (281, 476), (477, 580)],
   'd2': [(1, 140), (259, 409), (461, 560), (638, 730)],
   'd3': [(1, 155), (156, 255), (256, 402), (403, 552)],
   'd4': [(1, 128), (129, 278), (279, 432), (433, 595)]},
  'jogging': {'d1': [(1, 70), (80, 145), (220, 290), (315, 390)],
   'd2': [(1, 65), (112, 200), (270, 365), (430, 520)],
   'd3': [(1, 60), (100, 160), (230, 290), (370, 425)],
   'd4': [(1, 70), (108, 160), (380, 465), (500, 580)]},
  'running': {'d1': [(1, 50), (105, 155), (218, 270), (325, 375)],
   'd2': [(1, 50), (82, 130), (190, 250), (285, 350)],
   'd3': [(1, 45), (100, 140), (205, 250), (310, 350)],
   'd4': [(1, 50), (405, 465), (506, 556), (590, 666)]},
  'walking': {'d1': [(1, 95), (120, 210), (495, 585), (645, 745)],
   'd2': [(1, 120), (168, 275), (380, 495), (535, 645)],
   'd3': [(1, 95), (168, 255), (360, 450), (550, 640)],
   'd4': [(1, 80), (125, 210), (224, 320), (395, 492)]}},
 'person19': {'boxing': {'d1': [(1, 108), (109, 182), (183, 269), (270, 350)],
   'd2': [(1, 126), (127, 255), (256, 354), (355, 468)],
   'd3': [(1, 92), (93, 185), (186, 310), (311, 442)],
   'd4': [(1, 124), (125, 210), (211, 298), (299, 420)]},
  'handclapping': {'d1': [(1, 88), (89, 200), (201, 290), (291, 412)],
   'd2': [(1, 84), (85, 181), (230, 342), (343, 450)],
   'd3': [(1, 115), (116, 212), (213, 335), (336, 460)],
   'd4': [(1, 102), (103, 203), (204, 282), (283, 384)]},
  'handwaving': {'d1': [(1, 140), (141, 290), (291, 444), (445, 655)],
   'd2': [(1, 144), (212, 340), (341, 485), (486, 630)],
   'd3': [(1, 128), (129, 312), (313, 450), (451, 638)],
   'd4': [(1, 155), (156, 330), (331, 466), (467, 599)]},
  'jogging': {'d1': [(1, 60), (108, 164), (185, 245), (300, 365)],
   'd2': [(1, 85), (140, 245), (265, 355), (420, 525)],
   'd3': [(1, 50), (80, 135), (175, 225), (295, 350)],
   'd4': [(1, 70), (100, 180), (188, 250), (285, 350)]},
  'running': {'d1': [(1, 45), (98, 140), (205, 250), (310, 355)],
   'd2': [(1, 64), (130, 205), (260, 325), (395, 475)],
   'd3': [(1, 40), (105, 148), (232, 275), (325, 365)],
   'd4': [(1, 50), (85, 135), (160, 212), (252, 310)]},
  'walking': {'d1': [(1, 85), (125, 215), (285, 370), (435, 525)],
   'd2': [(1, 125), (200, 358), (405, 550), (655, 805)],
   'd3': [(1, 80), (136, 220), (298, 380), (442, 525)],
   'd4': [(1, 128), (188, 315), (324, 428), (452, 570)]}},
 'person20': {'boxing': {'d1': [(1, 112), (113, 206), (207, 300), (301, 390)],
   'd2': [(1, 118), (119, 218), (219, 366), (367, 478)],
   'd3': [(1, 130), (131, 290), (291, 424), (425, 496)],
   'd4': [(1, 95), (96, 200), (201, 290), (291, 420)]},
  'handclapping': {'d1': [(1, 130), (131, 242), (243, 360), (361, 422)],
   'd2': [(1, 132), (133, 249), (250, 335), (336, 427)],
   'd3': [(1, 125), (126, 225), (226, 360), (361, 505)],
   'd4': [(1, 128), (129, 262), (263, 358), (359, 486)]},
  'handwaving': {'d1': [(1, 125), (126, 265), (266, 400), (401, 535)],
   'd2': [(1, 108), (109, 254), (255, 365), (366, 510)],
   'd3': [(1, 115), (116, 232), (233, 350), (351, 510)],
   'd4': [(1, 114), (115, 270), (271, 385), (386, 502)]},
  'jogging': {'d1': [(1, 50), (185, 235), (286, 345), (410, 470)],
   'd2': [(1, 85), (125, 218), (255, 330), (370, 470)],
   'd3': [(1, 45), (95, 145), (200, 245), (304, 350)],
   'd4': [(1, 65), (96, 155), (156, 225), (260, 320)]},
  'running': {'d1': [(1, 30), (89, 120), (200, 235), (298, 330)],
   'd2': [(1, 50), (88, 150), (190, 240), (280, 345)],
   'd3': [(1, 35), (70, 100), (160, 190), (235, 270)],
   'd4': [(1, 40), (80, 125), (138, 190), (220, 265)]},
  'walking': {'d1': [(1, 90), (150, 258), (350, 445), (540, 635)],
   'd2': [(1, 162), (295, 455), (512, 655), (735, 888)],
   'd3': [(1, 80), (105, 180), (295, 370), (420, 495)],
   'd4': [(1, 105), (150, 245), (258, 355), (365, 480)]}},
 'person21': {'boxing': {'d1': [(1, 75), (76, 203), (204, 286), (287, 365)],
   'd2': [(1, 112), (113, 162), (163, 254), (255, 408)],
   'd3': [(1, 142), (143, 242), (258, 346), (347, 485)],
   'd4': [(1, 120), (121, 245), (246, 356), (357, 466)]},
  'handclapping': {'d1': [(1, 118), (119, 224), (225, 365), (366, 500)],
   'd2': [(1, 115), (116, 230), (231, 394), (395, 558)],
   'd3': [(1, 116), (117, 210), (211, 302), (303, 428)],
   'd4': [(1, 112), (155, 295), (296, 442), (443, 550)]},
  'handwaving': {'d1': [(1, 130), (131, 284), (285, 438), (439, 605)],
   'd2': [(1, 155), (156, 355), (356, 510), (511, 664)],
   'd3': [(1, 158), (159, 272), (273, 424), (425, 535)],
   'd4': [(1, 145), (146, 240), (330, 464), (465, 638)]},
  'jogging': {'d1': [(1, 45), (175, 225), (355, 402), (534, 580)],
   'd2': [(1, 100), (140, 232), (385, 486), (528, 620)],
   'd3': [(1, 45), (155, 200), (342, 385), (490, 535)],
   'd4': [(1, 80), (160, 224), (245, 305), (382, 450)]},
  'running': {'d1': [(1, 32), (115, 155), (262, 300), (392, 430)],
   'd2': [(1, 70), (112, 190), (325, 400), (436, 510)],
   'd3': [(1, 32), (115, 150), (275, 310), (390, 430)],
   'd4': [(1, 52), (110, 158), (178, 230), (285, 330)]},
  'walking': {'d1': [(1, 82), (188, 270), (436, 514), (648, 740)],
   'd2': [(1, 132), (195, 350), (550, 725), (768, 930)],
   'd3': [(1, 70), (195, 260), (506, 575), (700, 775)],
   'd4': [(1, 130), (172, 288), (298, 400), (490, 612)]}},
 'person22': {'boxing': {'d1': [(1, 92), (104, 208), (222, 302)],
   'd2': [(1, 74), (148, 248), (262, 320), (337, 417)],
   'd3': [(1, 99), (125, 226), (227, 340), (363, 428)],
   'd4': [(1, 80), (104, 184), (208, 300), (314, 435)]},
  'handclapping': {'d1': [(1, 102), (103, 178), (179, 250), (251, 344)],
   'd2': [(1, 115), (116, 228), (229, 338), (339, 471)],
   'd3': [(1, 90), (91, 180), (181, 300), (301, 422)],
   'd4': [(1, 122), (123, 224), (225, 346), (347, 440)]},
  'handwaving': {'d1': [(1, 128), (129, 258), (259, 440), (441, 525)],
   'd2': [(1, 175), (176, 260), (261, 392), (393, 528)],
   'd3': [(1, 88), (89, 176), (224, 355), (356, 486)],
   'd4': [(1, 155), (156, 292), (293, 382), (383, 512)]},
  'jogging': {'d1': [(1, 45), (145, 192), (252, 306), (436, 496)],
   'd2': [(1, 85), (142, 230), (262, 352), (416, 502)],
   'd3': [(1, 50), (162, 212), (288, 338), (455, 510)],
   'd4': [(1, 64), (102, 166), (188, 250), (285, 350)]},
  'running': {'d1': [(1, 32), (130, 164), (212, 250), (344, 375)],
   'd2': [(1, 55), (128, 180), (240, 295), (378, 425)],
   'd3': [(1, 32), (134, 168), (292, 328), (425, 465)],
   'd4': [(1, 50), (98, 150), (162, 208), (262, 310)]},
  'walking': {'d1': [(1, 102), (202, 310), (372, 468), (586, 700)],
   'd2': [(1, 152), (225, 375), (465, 618), (760, 920)],
   'd3': [(1, 100), (258, 350), (495, 580), (720, 815)],
   'd4': [(1, 110), (130, 248), (252, 372), (412, 540)]}},
 'person23': {'boxing': {'d1': [(1, 95), (96, 172), (258, 372), (373, 482)],
   'd2': [(1, 128), (129, 250), (251, 392), (393, 500)],
   'd3': [(1, 82), (83, 218), (219, 340), (341, 448)],
   'd4': [(1, 115), (116, 247), (248, 316), (317, 368)]},
  'handclapping': {'d1': [(1, 126), (127, 222), (223, 346), (347, 440)],
   'd2': [(1, 100), (101, 235), (236, 366), (367, 458)],
   'd3': [(1, 142), (143, 258), (259, 370), (371, 482)],
   'd4': [(1, 118), (119, 266), (267, 340), (341, 444)]},
  'handwaving': {'d1': [(1, 126), (127, 256), (257, 392), (393, 576)],
   'd2': [(1, 122), (123, 244), (245, 370), (535, 658)],
   'd3': [(1, 122), (123, 284), (285, 366), (367, 484)],
   'd4': [(1, 136), (137, 266), (267, 436), (437, 526)]},
  'jogging': {'d1': [(1, 46), (108, 154), (225, 272), (318, 368)],
   'd2': [(1, 104), (120, 238), (264, 374), (404, 528)],
   'd3': [(1, 45), (72, 120), (190, 232), (256, 302)],
   'd4': [(1, 60), (68, 132), (168, 230), (240, 310)]},
  'running': {'d1': [(1, 26), (96, 122), (212, 242), (314, 342)],
   'd2': [(1, 64), (88, 160), (170, 240), (265, 332)],
   'd3': [(1, 32), (70, 102), (172, 200), (250, 280)],
   'd4': [(1, 50), (58, 100), (102, 148), (164, 204)]},
  'walking': {'d1': [(1, 80), (156, 250), (346, 440), (498, 590)],
   'd2': [(1, 174), (230, 426), (442, 634), (672, 850)],
   'd3': [(1, 84), (128, 212), (342, 424), (480, 555)],
   'd4': [(1, 94), (134, 234), (242, 330), (380, 478)]}},
 'person24': {'boxing': {'d1': [(1, 112), (113, 190), (191, 280), (281, 366)],
   'd2': [(1, 122), (123, 246), (275, 348), (349, 452)],
   'd3': [(1, 94), (95, 188), (189, 294), (295, 390)],
   'd4': [(1, 110), (111, 190), (191, 268), (269, 382)]},
  'handclapping': {'d1': [(1, 94), (95, 192), (193, 310), (311, 380)],
   'd2': [(1, 92), (93, 208), (209, 280), (281, 385)],
   'd3': [(1, 90), (91, 200), (201, 314), (315, 402)],
   'd4': [(1, 86), (87, 194), (195, 262), (263, 350)]},
  'handwaving': {'d1': [(1, 138), (139, 270), (271, 454), (455, 588)],
   'd2': [(1, 140), (141, 278), (279, 417), (418, 510)],
   'd3': [(1, 126), (127, 258), (259, 345), (346, 520)],
   'd4': [(1, 134), (135, 265), (266, 352), (353, 525)]},
  'jogging': {'d1': [(1, 55), (166, 222), (390, 448), (518, 570)],
   'd2': [(1, 108), (168, 272), (378, 480), (545, 655)],
   'd3': [(1, 55), (128, 184), (250, 310), (392, 452)],
   'd4': [(1, 82), (94, 175), (185, 265), (295, 375)]},
  'running': {'d1': [(1, 32), (108, 140), (226, 260), (340, 376)],
   'd2': [(1, 88), (154, 234), (330, 412), (472, 550)],
   'd3': [(1, 40), (110, 150), (228, 270), (354, 392)],
   'd4': [(1, 55), (90, 148), (156, 212), (256, 308)]},
  'walking': {'d1': [(1, 80), (172, 248), (420, 500), (592, 664)],
   'd2': [(1, 184), (272, 448), (544, 712), (797, 960)],
   'd3': [(1, 80), (165, 245), (324, 402), (500, 580)],
   'd4': [(1, 88), (104, 196), (206, 290), (305, 395)]}},
 'person25': {'boxing': {'d1': [(1, 108), (109, 195), (196, 305), (306, 396)],
   'd2': [(1, 108), (109, 218), (219, 322), (323, 456)],
   'd3': [(1, 130), (131, 244), (245, 394), (395, 474)],
   'd4': [(1, 120), (121, 224), (225, 326), (327, 460)]},
  'handclapping': {'d1': [(1, 104), (105, 208), (209, 340), (341, 418)],
   'd2': [(1, 124), (125, 210), (211, 364), (365, 485)],
   'd3': [(1, 106), (107, 188), (189, 323), (324, 438)],
   'd4': [(1, 126), (127, 284), (285, 374), (375, 500)]},
  'handwaving': {'d1': [(1, 120), (121, 280), (281, 360), (361, 475)],
   'd2': [(1, 110), (111, 260), (261, 372), (373, 484)],
   'd3': [(1, 130), (131, 300), (301, 384), (385, 510)],
   'd4': [(1, 118), (119, 274), (275, 358), (359, 480)]},
  'jogging': {'d1': [(1, 58), (82, 138), (148, 200), (235, 284)],
   'd2': [(1, 95), (120, 210), (272, 365), (392, 484)],
   'd3': [(1, 62), (86, 148), (176, 232), (258, 312)],
   'd4': [(1, 70), (126, 190), (196, 260), (315, 380)]},
  'running': {'d1': [(1, 42), (70, 108), (124, 165), (192, 232)],
   'd2': [(1, 72), (90, 160), (192, 270), (286, 360)],
   'd3': [(1, 48), (66, 110), (136, 182), (200, 248)],
   'd4': [(1, 55), (92, 140), (142, 195), (240, 294)]},
  'walking': {'d1': [(1, 100), (162, 256), (300, 400), (474, 572)],
   'd2': [(1, 162), (192, 350), (454, 632), (640, 800)],
   'd3': [(1, 106), (132, 235), (280, 380), (410, 520)],
   'd4': [(1, 100), (178, 295), (296, 398), (488, 592)]}}}