
from prepare_datav2 import Data
from model import Model
from data.maluuba.maluubadata import MaluubaData
from data.cornell.cornelldata import CornellData
from data.portugues.portuguesdata import PortuguesData
from data.alexa.alexadata import AlexaData

# import msvcrt

# def main():
    
#     __exit = False
#     while True:
#         print("Press [ENTER] - to show this message")
#         r = msvcrt.getch()
#         r = r.upper()
#         if r == b'\x1b':
#             print("EXIT")
#             break
#         elif 
#         # elif r == 'b\r':
#         #     print("Press [ENTER] - to show this message")

def train():
    print()
    print("Train Module")
    data = PortuguesData()
    data.createDataset()
    # exit()
    model = Model()
    model.load(data)
    model.train(epochs=1000)

def interact():
    print("Interact Module")
    
    data = PortuguesData()
    data.loadVocab()
    model = Model()
    model.load(data,with_checkpoint=True)

    while True:
        print()
        message = input("Você: ")
        response = model.get_response(message)
        print()
        print("Bot: "+ response[1])
        if not response[0]:
            break
    print()

if __name__ == '__main__':
    import argparse

    ACTION_CHOICES = ('train', 'interact')
    parser = argparse.ArgumentParser()

    parser.add_argument('-a', '--action', choices=ACTION_CHOICES, default='interact', help="Action to perform {}".format(ACTION_CHOICES))
    args = parser.parse_args()

    if args.action == 'train':
        train()
    elif args.action == 'interact':
        interact()
