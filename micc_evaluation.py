#   This file is subject to the terms and conditions defined in file 'LICENSE.txt',
#   which is part of this source code package.

""" Run MICC Florence dataset Experiment
    Author: Baris Gecer """
import os
import argparse
import numpy as np

import menpo3d.io as m3io
import menpo.io as mio
from menpo.shape import PointCloud, TriMesh
from menpo.transform import AlignmentSimilarity
from menpo3d.vtkutils import trimesh_to_vtk, VTKClosestPointLocator


class MICC_Benchmark():
    def __init__(self, args):
        # Paths
        self.landmarks_path = args.landmarks
        self.registration_path = args.registration_path

        # Load Template
        self.template = m3io.import_mesh(args.template)
        self.template_lms = np.array(mio.import_pickle(args.template_lms))
        self.template.landmarks['LJSON'] = PointCloud(self.template.points[self.template_lms])


    def point2plane_ICP(self, reg, rec, tight_mask):

        if tight_mask:
            mask = (np.sum((reg.points - reg.landmarks['LJSON'].points[30])**2,1)**0.5)<0.6
        else:
            mask = (np.sum((reg.points - reg.landmarks['LJSON'].points[30])**2,1)**0.5)<0.95

        target_vtk = trimesh_to_vtk(rec)
        closest_points_on_target = VTKClosestPointLocator(target_vtk)

        rec_closest = rec.from_mask(mask).points

        dist = np.mean(np.sum((rec_closest - reg.from_mask(mask).points)**2,1)**0.5)*100
        pre_dist=np.Inf
        print('Running ICP:', end='')
        while pre_dist - dist > 0.001:
            lm_align = AlignmentSimilarity(PointCloud(reg.from_mask(mask).points), PointCloud(rec_closest)).as_non_alignment()
            reg = lm_align.apply(reg)

            rec_closest, tri_indices = closest_points_on_target(reg.from_mask(mask).points)
            pre_dist = dist
            dist = np.mean(np.sum((rec_closest - reg.from_mask(mask).points)**2,1)**0.5)*100
            print('.',end='')
        print('Point-to-Plane Distance: ' + "%0.02f"%(dist))
        return dist

    def calculate_error(self, fitting_path, gt_path, id, setting, scan, tight_mask=False):
        org = m3io.import_meshes(self.registration_path + gt_path + 'obj/*.obj')[0]
        org_lms = m3io.import_landmark_files(self.landmarks_path + gt_path + 'obj/*.ljson')[0]['LJSON']
        org.landmarks['LJSON'] = org_lms

        lm_align = AlignmentSimilarity(org.landmarks['LJSON'], self.template.landmarks['LJSON']).as_non_alignment()
        org = lm_align.apply(org)

        reg = m3io.import_meshes(self.registration_path + gt_path +'obj/*.obj')[0]
        reg.landmarks = org.landmarks

        rec = m3io.import_mesh(fitting_path + '/subject_' + "%02d" % (id) + '/' + setting + '.obj')
        rec.landmarks['LJSON'] = PointCloud(rec.points[self.template_lms])

        return (self.point2plane_ICP(reg,rec,tight_mask) + self.point2plane_ICP(rec,reg,tight_mask))/2

    def benchmark(self, fitting_path):
        distances = np.ones([3,2,53]) * np.Inf
        for setting_id, setting in enumerate(['Indoor-Cooperative','PTZ-Indoor','PTZ-Outdoor']):
            for scan in [1,2]:
                for id in range(1,54):
                    gt_path = '/subject_'+"%02d"%(id)+'/Model/frontal'+ str(scan) +'/'
                    if os.path.exists(self.landmarks_path + gt_path):
                        print('ID: ' + str(id))

                        if id ==28:
                            distances[setting_id, scan-1, id-1] = self.calculate_error(fitting_path, gt_path, id, setting, scan, True)
                        else:
                            distances[setting_id, scan-1, id-1] = self.calculate_error(fitting_path, gt_path, id, setting, scan, False)

        return np.min(np.array(distances),1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument("registration_path", help="path to the registered MICC Florence dataset")
    parser.add_argument("reconstruction_path", help="path to the reconstructions to be evaluated")
    parser.add_argument('--landmarks', default='MICC_landmarks', help='Manually labelled ibug68 landmarks of the MICC dataset')
    parser.add_argument('--template', default='template.obj', help='Template face obj')
    parser.add_argument('--template_lms', default='landmark_ids.pkl', help='Landmark indices of the template Trimesh as pkl file')
    args, _ = parser.parse_known_args()

    benchmarker = MICC_Benchmark(args)

    distances = benchmarker.benchmark(args.reconstruction_path)

    mio.export_pickle(distances, os.path.join(args.reconstruction_path, 'distances.pkl'))
    means = np.mean(distances,1)
    stds = np.std(distances, 1)
    with open(os.path.join(args.reconstruction_path, "scores.txt"), "w") as text_file:
        text_file.write("Indoor-Cooperative : %.3f ± %.3f \n" % (means[0], stds[0]))
        text_file.write("PTZ-Indoor         : %.3f ± %.3f  \n" % (means[1], stds[1]))
        text_file.write("PTZ-Outdoor        : %.3f ± %.3f  \n" % (means[2], stds[2]))



