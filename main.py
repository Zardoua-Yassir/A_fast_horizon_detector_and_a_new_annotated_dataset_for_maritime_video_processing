from FastHorizonAlg import FastHorizon


def main():
    fast_horizon_algorithm = FastHorizon()  # instantiate the algorithm
    # demo on a video sample
    fast_horizon_algorithm.video_demo(video_path="./video sample/MMD_annotated_5.avi", display=True)


if __name__ == '__main__':
    main()
