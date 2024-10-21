import torch 

mapping = {}
act_shapes = {}

mapping['humanoid'] = {
    'torso': (list(range(10)), []),
    'head': ([], []),
    'lower_waist': (list(range(10, 14)), list(range(2))),
    'pelvis': (list(range(14, 16)), list(range(2,3))),
    'right_thigh': (list(range(16, 22)), list(range(3, 6))),
    'right_shin': (list(range(22, 24)), list(range(6, 7))),
    'right_foot': (list(range(24, 28)), list(range(7, 9))),
    'left_thigh': (list(range(28, 34)), list(range(9, 12))),
    'left_shin': (list(range(34, 36)), list(range(12, 13))),
    'left_foot': (list(range(36, 40)), list(range(13, 15))),
    'right_upper_arm': (list(range(40, 44)), list(range(15, 17))),
    'right_lower_arm': (list(range(44, 46)), list(range(17, 18))),
    'right_hand': ([], []),
    'left_upper_arm': (list(range(46, 50)), list(range(18, 20))),
    'left_lower_arm': (list(range(50, 52)), list(range(20, 21))),
    'left_hand': ([], []),
}

# mapping['humanoid_with_forces'] = {
#     'map': {
#         'torso': (list(range(10)) + list(range(52, 55)), []),
#         'head': (list(range(55, 58)), []),
#         'lower_waist': (list(range(10, 14)) + list(range(58, 61)), list(range(2))),
#         'pelvis': (list(range(14, 16)) + list(range(61, 64)), list(range(2,3))),
#         'right_thigh': (list(range(16, 22)) + list(range(64, 67)), list(range(3, 6))),
#         'right_shin': (list(range(22, 24)) + list(range(67, 70)), list(range(6, 7))),
#         'right_foot': (list(range(24, 28)) + list(range(70, 73)), list(range(7, 9))),
#         'left_thigh': (list(range(28, 34)) + list(range(73, 76)), list(range(9, 12))),
#         'left_shin': (list(range(34, 36)) + list(range(76, 79)), list(range(12, 13))),
#         'left_foot': (list(range(36, 40)) + list(range(79, 82)), list(range(13, 15))),
#         'right_upper_arm': (list(range(40, 44)) + list(range(82, 85)), list(range(15, 17))),
#         'right_lower_arm': (list(range(44, 46)) + list(range(85, 88)), list(range(17, 18))),
#         'right_hand': (list(range(88, 91)), []),
#         'left_upper_arm': (list(range(46, 50)) + list(range(91, 94)), list(range(18, 20))),
#         'left_lower_arm': (list(range(50, 52)) + list(range(94, 97)), list(range(20, 21))),
#         'left_hand': (list(range(97, 100)), []),
#     },
#     'sp_matrix': torch.Tensor(
#         [[0, 1, 1, 2, 3, 4, 5, 3, 4, 5, 1, 2, 3, 1, 2, 3],
#         [1, 0, 2, 3, 4, 5, 6, 4, 5, 6, 2, 3, 4, 2, 3, 4],
#         [1, 2, 0, 1, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4],
#         [2, 3, 1, 0, 1, 2, 3, 1, 2, 3, 3, 4, 5, 3, 4, 5],
#         [3, 4, 2, 1, 0, 1, 2, 2, 3, 4, 4, 5, 6, 4, 5, 6],
#         [4, 5, 3, 2, 1, 0, 1, 3, 4, 5, 5, 6, 7, 5, 6, 7],
#         [5, 6, 4, 3, 2, 1, 0, 4, 5, 6, 6, 7, 8, 6, 7, 8],
#         [3, 4, 2, 1, 2, 3, 4, 0, 1, 2, 4, 5, 6, 4, 5, 6],
#         [4, 5, 3, 2, 3, 4, 5, 1, 0, 1, 5, 6, 7, 5, 6, 7],
#         [5, 6, 4, 3, 4, 5, 6, 2, 1, 0, 6, 7, 8, 6, 7, 8],
#         [1, 2, 2, 3, 4, 5, 6, 4, 5, 6, 0, 1, 2, 2, 3, 4],
#         [2, 3, 3, 4, 5, 6, 7, 5, 6, 7, 1, 0, 1, 3, 4, 5],
#         [3, 4, 4, 5, 6, 7, 8, 6, 7, 8, 2, 1, 0, 4, 5, 6],
#         [1, 2, 2, 3, 4, 5, 6, 4, 5, 6, 2, 3, 4, 0, 1, 2],
#         [2, 3, 3, 4, 5, 6, 7, 5, 6, 7, 3, 4, 5, 1, 0, 1],
#         [3, 4, 4, 5, 6, 7, 8, 6, 7, 8, 4, 5, 6, 2, 1, 0]]
#     )
# }

mapping['humanoid_with_forces'] = {
    'map': {
        'torso': (list(range(10)) + list(range(52, 55)), []),
        'head': (list(range(55, 58)), []),
        'lower_waist': (list(range(10, 12)) + list(range(31, 33)) + list(range(58, 61)), list(range(2))),
        'pelvis': (list(range(12, 13)) + list(range(33, 34)) + list(range(61, 64)), list(range(2,3))),
        'right_thigh': (list(range(13, 16)) + list(range(34, 37)) + list(range(64, 67)), list(range(3, 6))),
        'right_shin': (list(range(16, 17)) + list(range(37, 38)) + list(range(67, 70)), list(range(6, 7))),
        'right_foot': (list(range(17, 19)) + list(range(38, 40)) + list(range(70, 73)), list(range(7, 9))),
        'left_thigh': (list(range(19, 22)) + list(range(40, 43)) + list(range(73, 76)), list(range(9, 12))),
        'left_shin': (list(range(22, 23)) + list(range(43, 44)) + list(range(76, 79)), list(range(12, 13))),
        'left_foot': (list(range(23, 25)) + list(range(44, 46)) + list(range(79, 82)), list(range(13, 15))),
        'right_upper_arm': (list(range(25, 27)) + list(range(46, 48)) + list(range(82, 85)), list(range(15, 17))),
        'right_lower_arm': (list(range(27, 28)) + list(range(48, 49)) + list(range(85, 88)), list(range(17, 18))),
        'right_hand': (list(range(88, 91)), []),
        'left_upper_arm': (list(range(28, 30)) + list(range(49, 51)) + list(range(91, 94)), list(range(18, 20))),
        'left_lower_arm': (list(range(30, 31)) + list(range(51, 52)) + list(range(94, 97)), list(range(20, 21))),
        'left_hand': (list(range(97, 100)), []),
    },
    'sp_matrix': torch.Tensor(
        [[0, 1, 1, 2, 3, 4, 5, 3, 4, 5, 1, 2, 3, 1, 2, 3],
        [1, 0, 2, 3, 4, 5, 6, 4, 5, 6, 2, 3, 4, 2, 3, 4],
        [1, 2, 0, 1, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4],
        [2, 3, 1, 0, 1, 2, 3, 1, 2, 3, 3, 4, 5, 3, 4, 5],
        [3, 4, 2, 1, 0, 1, 2, 2, 3, 4, 4, 5, 6, 4, 5, 6],
        [4, 5, 3, 2, 1, 0, 1, 3, 4, 5, 5, 6, 7, 5, 6, 7],
        [5, 6, 4, 3, 2, 1, 0, 4, 5, 6, 6, 7, 8, 6, 7, 8],
        [3, 4, 2, 1, 2, 3, 4, 0, 1, 2, 4, 5, 6, 4, 5, 6],
        [4, 5, 3, 2, 3, 4, 5, 1, 0, 1, 5, 6, 7, 5, 6, 7],
        [5, 6, 4, 3, 4, 5, 6, 2, 1, 0, 6, 7, 8, 6, 7, 8],
        [1, 2, 2, 3, 4, 5, 6, 4, 5, 6, 0, 1, 2, 2, 3, 4],
        [2, 3, 3, 4, 5, 6, 7, 5, 6, 7, 1, 0, 1, 3, 4, 5],
        [3, 4, 4, 5, 6, 7, 8, 6, 7, 8, 2, 1, 0, 4, 5, 6],
        [1, 2, 2, 3, 4, 5, 6, 4, 5, 6, 2, 3, 4, 0, 1, 2],
        [2, 3, 3, 4, 5, 6, 7, 5, 6, 7, 3, 4, 5, 1, 0, 1],
        [3, 4, 4, 5, 6, 7, 8, 6, 7, 8, 4, 5, 6, 2, 1, 0]]
    )
}

# mapping['actions'] = {
#     '0': [],
#     '1': [],
#     '2': [0, 1],
#     '3': [2],
#     '4': [3, 4, 5],
#     '5': [6],
#     '6': [7, 8],
#     '7': [9, 10, 11],
#     '8': [12],
#     '9': [13, 14],
#     '10': [15, 16],
#     '11': [17],
#     '12': [],
#     '13': [18, 19],
#     '14': [20]
# }

def get_mapping(state_type):
    return mapping[state_type]
