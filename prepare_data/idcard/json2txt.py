import json
import argparse
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

TXT_FORMAT= '{im_path} {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}\n'

def json2txt(jsonFileIn, txtFileOut):
    txtfile = open(txtFileOut, 'w')
    for jsonfile_path in jsonfileIn:
        jsonfile = open(jsonfile_path)
        jsonfile = jsonfile.readlines()

        jsonfile = jsonfile[0].strip()
        jsoncontent = json.loads(jsonfile)
        #num_imgs = jsoncontent[0]
        #assert num_imgs == len(jsoncontent[1]), "num of images:{} is not equal to json record number:{}".format(num_imgs, len(jsoncontent[1]))

        for record in jsoncontent[1]:
            im_path = record['path']
            im_scale = record['scale']
            pts = record['pts']
            if len(pts) != 4:
                print "num of pts != 4: {}-{}".format(im_path, pts)
                continue
            print pts, im_scale
            print pts[3][0], pts[3][1]
            line = TXT_FORMAT.format(im_path = im_path,
                                     x1 = pts[0][0] / im_scale, y1 = pts[0][1] / im_scale,
                                     x2 = pts[1][0] / im_scale, y2 = pts[1][1] / im_scale,
                                     x3 = pts[2][0] / im_scale, y3 = pts[2][1] / im_scale,
                                     x4 = pts[3][0] / im_scale, y4 = pts[3][1] / im_scale)
            print(line)
            txtfile.write(line)
    txtfile.close()

def parse_args():
    parser = argparse.ArgumentParser(description='idcard info')

    parser.add_argument('--jsonfileIn', help='idcard annotation in json format, support multiple jsonfile',
                        type=str)
    parser.add_argument('--txtfileOut', help='idcard annotation in txt format', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)
    jsonfileIn = args.jsonfileIn
    if ',' in args.jsonfileIn:
        jsonfileIn = args.jsonfileIn.strip().split(',')

    json2txt(jsonfileIn, args.txtfileOut)


