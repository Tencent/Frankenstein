import json
import numpy as np 
import matplotlib.pyplot as plt 
import os

# RGB for visualization
COLOR_MAP_20 = {
    -1: (0., 0., 0.), 
    0: (174., 199., 232.), 
    1: (152., 223., 138.), 
    2: (31., 119., 180.), 
    3: (255., 187., 120.), 
    4: (188., 189., 34.), 
    5: (140., 86., 75.),
    6: (255., 152., 150.), 
    7: (214., 39., 40.), 
    8: (197., 176., 213.), 
    9: (148., 103., 189.),
}

# visualize layout images by painting different semantic colors
def img2vis(img, sem_info):
    # sort by area (avoid occlusion)
    sem_info = sorted(sem_info, key=lambda x: (x[1][0]-x[0][0])*(x[1][2]-x[0][2]), reverse=True)
    color_img = np.zeros([img.shape[0], img.shape[1], 3])
    for info in sem_info:
        aa,bb,sem_i = info
        color_img[aa[0]:bb[0]+1, aa[2]:bb[2]+1] = np.array(COLOR_MAP_20[sem_i]) / 255.
    return color_img



# begin to create layouts
def paint_layout(args):
    # path to save layout maps
    savedir = os.path.join(f"painting_{args.roomtype}", args.baseroom)
    os.makedirs(savedir, exist_ok=True)

    # choose baseroom and load bbox info
    with open(f"diff_denoising/semimg_{args.roomtype}.json", 'r') as f:
        info = json.load(f)
    data = info[args.baseroom]

    # print room info
    # bedroom has 3 classes, layout map is 40x40x3
    if args.roomtype == "bedroom":
        reso = 40
        n_labels = 3
        print(
            f"{args.baseroom} has "
            f"{len(data['0'])} walls, "
            f"{len(data['1'])} beds and "
            f"{len(data['2'])} cabinets."
        )
    # livingroom has 5 classes, layout map is 56x56x5
    elif args.roomtype == "livingroom":
        reso = 56
        n_labels = 5
        print((
            f"{args.baseroom} has "
            f"{len(data['0'])} walls, "
            f"{len(data['1'])} cabinets, "
            f"{len(data['2'])} chairs, "
            f"{len(data['3'])} sofas and "
            f"{len(data['4'])} tables."
        ))
    else:
        # pre-trained models only support bedroom and livingroom 
        raise NotImplementedError

    assert len([data['0']]) == 1, "Only one wall is allowed"

    # determine the valid range for moving
    min_bound = data['0'][0][0][args.direction]
    max_bound = data['0'][0][1][args.direction]

    objinfo = data[args.semid][args.objid]
    ori_a = objinfo[0][args.direction]
    ori_b = objinfo[1][args.direction]
    p = int((ori_a - min_bound) / args.stride)
    q = int((max_bound - ori_b) / args.stride)

    with open(os.path.join(savedir, "paint_info.json"), 'w') as f:
        f.write(json.dumps({
            'stride': args.stride,
            'direction': args.direction,
            'semid': args.semid,
            'objid': args.objid,
            'range': [-p,q]
        }, indent=4))

    plt.figure(1)
    all_layouts = []
    # move object and generate p+q layout maps
    for n in range(-p, q+1):
        objinfo[0][args.direction] = ori_a + n * args.stride
        objinfo[1][args.direction] = ori_b + n * args.stride
        XZ_img = np.zeros([reso, reso, n_labels], dtype=np.float32)
        sem_info = [] # (aa,bb,i)
        for i in range(n_labels):
            sem = np.eye(n_labels)[i]

            for aa,bb in data[str(i)]:
                aa = np.array(aa)
                bb = np.array(bb)
                aa = np.round((aa+1)/2*(reso-1)).astype(np.int64)
                bb = np.round((bb+1)/2*(reso-1)).astype(np.int64)
                XZ_img[aa[0]:bb[0]+1, aa[2]:bb[2]+1] += sem
                sem_info.append((aa,bb,i))

        XZ_img[XZ_img>1] = 1
        all_layouts.append(XZ_img)

        # check if the generated layout maps are reasonable
        plt.imshow(img2vis(XZ_img, sem_info), origin="lower")
        plt.savefig(os.path.join(savedir, f"{args.baseroom}_{p+n:03d}.png") )
        plt.clf()

    all_layouts = np.stack(all_layouts)
    print(f"Generated layout maps with shape: {all_layouts.shape}")
    np.save(os.path.join(savedir, "layout_maps.npy"), all_layouts)




if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--baseroom", "-b", type=str, required=True,
        help="room name (chosen from diff_denoising/semimg_*.json)",
    )
    arg_parser.add_argument(
        "--roomtype", "-t", type=str, required=True,
        help="bedroom/livingroom",
    )
    arg_parser.add_argument(
        "--stride", "-s", type=float, default=0.05,
        help="",
    )
    arg_parser.add_argument(
        "--direction", "-d", type=int, default=0,
        help="0 for x-axis, 2 for z-axis",
    )
    arg_parser.add_argument(
        "--semid", "-i", type=str, default='1',
        help="[bedroom] 1:bed 2:cabinet / [livingroom] 1: cabinet 2:chair 3:sofa 4:table",
    )
    arg_parser.add_argument(
        "--objid", "-o", type=int, default=0,
        help="",
    )

    args = arg_parser.parse_args()

    paint_layout(args)
