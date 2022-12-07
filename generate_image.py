from CycleGAN import CycleGAN
import Config


###
### Run train.py to get a trained model before running this script.
###

def main():
    model = CycleGAN()
    model.load_model()

    input_image = "summer.jpg"
    output_image = "winter.jpg"

    model.generate_img(input_image, output_image, reverse=False)


if __name__ == '__main__':
    main()
