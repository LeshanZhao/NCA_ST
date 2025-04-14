
import numpy as np
import cupy as cp
import time
class ripleyk:

    # (x_max, y_max, x_min, y_min) defines the rectangle area of interest where we used to calculate ripley's k value.
    # Not required for cell-wise methods (evaluate_bivariate_cell_wise() and evaluate_bivariate_cell_wise()). Use 0 values to initialize them.)

    def __init__(self, x_max=None, y_max=None, x_min=None, y_min=None):
        # print("RipleyK initialized")
        self.x_max = x_max
        self.y_max = y_max
        self.x_min = x_min
        self.y_min = y_min
        self.x_min_0 = self.x_min
        self.x_max_0 = self.x_max
        self.y_min_0 = self.y_min
        self.y_max_0 = self.y_max
        self.origional_boundary = [self.x_min_0,  self.x_max_0, self.y_min_0, self.y_max_0]
        self.area = (self.x_max - self.x_min) * (self.y_max - self.y_min)

    # Not useful until the last version of our package. Don't use it until we are ready.
    def __call__(self, data, radii, mode="none"):
        return self.evaluate(data=data, radii=radii, mode=mode)
    
    # Utility Internal function
    def _pairwise_diffs(self, data):
        npts = len(data)
        diff = np.zeros(shape=(npts * (npts - 1) // 2, 2), dtype=np.double)
        k = 0
        for i in range(npts - 1):
            size = npts - i - 1
            diff[k : k + size] = abs(data[i] - data[i + 1 :])
            k += size

        return diff

    def poisson(self, radii):
        """
        Evaluates the Ripley K function for the homogeneous Poisson process,
        also known as Complete State of Randomness (CSR).

        Parameters
        ----------
        radii : 1D array
            Set of distances in which Ripley's K function will be evaluated.

        Returns
        -------
        output : 1D array
            Ripley's K function evaluated at ``radii``.
        """
        return np.pi * radii * radii

    def Lfunction(self, data, radii, mode="none"):
        """
        Evaluates the L function at ``radii``. For parameter description
        see ``evaluate`` method.
        """
        return np.sqrt(self.evaluate(data, radii, mode=mode) / np.pi)

    def Hfunction(self, data, radii, mode="none"):
        """
        Evaluates the H function at ``radii``. For parameter description
        see ``evaluate`` method.
        """
        return self.Lfunction(data, radii, mode=mode) - radii

    # Utility Internal function
    def divide_boundaries(self, sub_size, n_sections):
        # window_length = self.x_max - self.x_min
        # window_width = self.y_max - self.y_min
        # if sub_size != 100:
        #     n_sections = 700 / sub_size
        if n_sections != 7:
            sub_size = 700 / n_sections
        if sub_size != 100:
            if 700 % sub_size != 0:
                raise ValueError(
                    "700 must be an integer multiplication of sub_size."
                )
            n_sections = int(700 / sub_size)
        
        boundaries = []
        n_blocks = n_sections * n_sections
        for block_i in range(n_blocks):
            x_min = self.x_min_0 + sub_size * (block_i % n_sections)
            x_max = x_min + sub_size
            y_min = self.y_min_0 + sub_size * (block_i // n_sections)
            y_max = y_min + sub_size
            boundaries.append([x_min, x_max, y_min, y_max])

        return boundaries, sub_size, n_sections

    # Utility Internal function
    def divide_data(self, data, boundaries, sub_size, n_sections):
        list_of_sub_data = [[] for _ in range(len(boundaries))]
        x_0, x_1, y_0, y_1 = boundaries[0]
        for i,row in enumerate(data):
            x = row[0]
            y = row[1]
            x_index = int( (x - x_0) // sub_size )
            y_index = int( (y - y_0) // sub_size )
            index = y_index * n_sections + x_index

            list_of_sub_data[index].append(row)

        divided_data = [np.array(sublist) for sublist in list_of_sub_data]

        return divided_data

    # Method 6
    def devide_n_conquer(self, data, radii, mode="none", sub_size=100, n_sections=7):
        boundaries, sub_size, n_sections = self.divide_boundaries(sub_size, n_sections)
        divided_data = self.divide_data(data, boundaries, sub_size, n_sections)
        # radii = list(range(1, int(sub_size//3)))
        print(radii)
        results = []
        for i,boundary in enumerate(boundaries):
            self.x_min, self.x_max, self.y_min, self.y_max = boundary
            if len(divided_data[i]) > 1:
                result = self.evaluate(divided_data[i], radii, mode=mode)
                results.append(result)
            else:
                result = np.zeros(len(radii))
                results.append(result)
        
        results = np.asarray(results)
            
        average_result = results.mean(axis=0)

        self.x_min, self.x_max, self.y_min, self.y_max = self.origional_boundary

        return average_result

    # Method 1
    def evaluate(self, data, radii, mode="none"):
        print("evalulating, mode:", mode)
        """
        Evaluates the Ripley K estimator for a given set of values ``radii``.

        Parameters
        ----------
        data : 2D array
            Set of observed points in as a n by 2 array which will be used to
            estimate Ripley's K function.
        radii : 1D array
            Set of distances in which Ripley's K estimator will be evaluated.
            Usually, it's common to consider max(radii) < (area/2)**0.5.
        mode : str
            Keyword which indicates the method for edge effects correction.
            Available methods are 'none' and 'ripley'.

            * 'none'
                this method does not take into account any edge effects
                whatsoever.
            * 'ripley'
                this method is known as Ripley's edge-corrected estimator.
                The weight for edge-correction is a function of the
                proportions of circumferences centered at each data point
                which crosses another data point of interest.
        Returns
        -------
        ripley : 1D array
            Ripley's K function estimator evaluated at ``radii``.
        """
        
        data = np.asarray(data)
        if not data.shape[1] == 2:
            raise ValueError(
                "data must be an n by 2 array, where n is the "
                "number of observed points."
            )

        npts = len(data)
        ripley = np.zeros(len(radii))

        if mode == "none":
            diff = self._pairwise_diffs(data)
            distances = np.hypot(diff[:, 0], diff[:, 1])
            for r in range(len(radii)):
                ripley[r] = (distances < radii[r]).sum()

            ripley = self.area * 2.0 * ripley / (npts * (npts - 1))
        # Ripley Edge correction
        elif mode == "ripley":
            hor_dist = np.zeros(shape=(npts * (npts - 1)) // 2, dtype=np.double)
            ver_dist = np.zeros(shape=(npts * (npts - 1)) // 2, dtype=np.double)

            for k in range(npts - 1):
                min_hor_dist = min(self.x_max - data[k][0], data[k][0] - self.x_min)
                min_ver_dist = min(self.y_max - data[k][1], data[k][1] - self.y_min)
                start = (k * (2 * (npts - 1) - (k - 1))) // 2
                end = ((k + 1) * (2 * (npts - 1) - k)) // 2
                hor_dist[start:end] = min_hor_dist * np.ones(npts - 1 - k)
                ver_dist[start:end] = min_ver_dist * np.ones(npts - 1 - k)

            diff = self._pairwise_diffs(data)
            dist = np.hypot(diff[:, 0], diff[:, 1])
            dist_ind = dist <= np.hypot(hor_dist, ver_dist)

            w1 = (
                1
                - (
                    np.arccos(np.minimum(ver_dist, dist) / dist)
                    + np.arccos(np.minimum(hor_dist, dist) / dist)
                )
                / np.pi
            )
            w2 = (
                3 / 4
                - 0.5
                * (
                    np.arccos(ver_dist / dist * ~dist_ind)
                    + np.arccos(hor_dist / dist * ~dist_ind)
                )
                / np.pi
            )

            weight = dist_ind * w1 + ~dist_ind * w2

            for r in range(len(radii)):
                ripley[r] = ((dist < radii[r]) / weight).sum()

            ripley = self.area * 2.0 * ripley / (npts * (npts - 1))
        else:
            raise ValueError(f"mode {mode} is not implemented.")

        return ripley
    
    # Method 2
    def evaluate_bivariate(self, data1, data2, radii, mode="none"):

        """
        Evaluates the bivariate Ripley's K function.

        Parameters
        ----------
        data1 : 2D array
            Set of observed points of type 1.
        data2 : 2D array
            Set of observed points of type 2.
        radii : 1D array
            Set of distances at which the bivariate K function will be evaluated.
        mode : str
            Mode for edge effect correction. Options are 'none' and 'ripley'.

        Returns
        -------
        bivariate_ripley : 1D array
            Bivariate Ripley's K function evaluated at `radii`.
        """
        if mode not in ["none", "ripley"]:
            raise ValueError(f"mode {mode} is not implemented.")

        data1 = np.asarray(data1)
        data2 = np.asarray(data2)
        npts1 = len(data1)
        npts2 = len(data2)
        bivariate_ripley = np.zeros(len(radii))

        if mode == "none":
            for r in range(len(radii)):
                for point in data1:
                    # distances = np.sqrt((data2[:, 0] - point[0]) ** 2 + (data2[:, 1] - point[1]) ** 2)
                    distances = np.hypot((data2[:, 0] - point[0]), (data2[:, 1] - point[1]))
                    bivariate_ripley[r] += (distances < radii[r]).sum()
            bivariate_ripley = self.area * bivariate_ripley / (npts1 * npts2)
        elif mode == "ripley":
            for r in range(len(radii)):
                for point in data1:
                    # distances = np.sqrt((data2[:, 0] - point[0]) ** 2 + (data2[:, 1] - point[1]) ** 2)
                    distances = np.hypot((data2[:, 0] - point[0]), (data2[:, 1] - point[1]))
                    min_hor_dist = min(self.x_max - point[0], point[0] - self.x_min)
                    min_ver_dist = min(self.y_max - point[1], point[1] - self.y_min)
                    max_dist = np.hypot(min_hor_dist, min_ver_dist)

                    # w = np.ones_like(distances)
                    mask = distances <= max_dist
                    w1 = (
                        1
                        - (
                            np.arccos(np.minimum(min_ver_dist, distances) / distances *mask)
                            + np.arccos(np.minimum(min_hor_dist, distances) / distances *mask)
                        )
                        / np.pi
                    )

                    w2 = (
                        3/4
                        - 0.5
                        * (
                            np.arccos(min_ver_dist / distances * ~mask)
                            + np.arccos(min_hor_dist / distances * ~mask)
                        )
                        / np.pi
                    )

                    w = mask * w1 + ~mask * w2

                    bivariate_ripley[r] += np.sum((distances < radii[r]) / w)

            bivariate_ripley = self.area * bivariate_ripley / (npts1 * npts2)

        return bivariate_ripley

    # Method 3
    def evaluate_weighted_bivariate(self, data1, data2, radii, mode="none"):
        npts1 = len(data1)
        npts2 = len(data2)

        lambda1 = npts1/self.area
        lambda2 = npts2/self.area

        return (lambda2 * self.evaluate_bivariate(data1, data2, radii, mode=mode) + 
                lambda1 * self.evaluate_bivariate(data2, data1, radii, mode=mode)) / (lambda1 + lambda2)
    
    # Method 4
    def evaluate_bivariate_extendMod(self, data1, ext_data2, npts2, radii, mode="extend"):

        """
        Evaluates the bivariate Ripley's K function.

        Parameters
        ----------
        data1 : 2D array
            Set of observed points of type 1.
        data2 : 2D array
            Set of observed points of type 2.
        radii : 1D array
            Set of distances at which the bivariate K function will be evaluated.
        mode : str
            Mode for edge effect correction. Options are 'none' and 'ripley'.

        Returns
        -------
        bivariate_ripley : 1D array
            Bivariate Ripley's K function evaluated at `radii`.
        """

        
        if mode not in ["none", "ripley", "extend", "merge_ext"]:
            raise ValueError(f"mode {mode} is not implemented.")

        data1 = np.asarray(data1)
        ext_data2 = np.asarray(ext_data2)
        npts1 = len(data1)
        # npts2 = len(ext_data2)
        bivariate_ripley = np.zeros(len(radii))
        if len(ext_data2) == 0 or npts1 == 0:
            return bivariate_ripley
    
        if mode == "none":
            for r in range(len(radii)):
                for point in data1:
                    # distances = np.sqrt((ext_data2[:, 0] - point[0]) ** 2 + (ext_data2[:, 1] - point[1]) ** 2)
                    distances = np.hypot((ext_data2[:, 0] - point[0]), (ext_data2[:, 1] - point[1]))
                    bivariate_ripley[r] += (distances < radii[r]).sum()
            bivariate_ripley = self.area * bivariate_ripley / (npts1 * npts2)
        elif mode == "ripley":
            for r in range(len(radii)):
                for point in data1:
                    # distances = np.sqrt((ext_data2[:, 0] - point[0]) ** 2 + (ext_data2[:, 1] - point[1]) ** 2)
                    distances = np.hypot((ext_data2[:, 0] - point[0]), (ext_data2[:, 1] - point[1]))
                    min_hor_dist = min(self.x_max - point[0], point[0] - self.x_min)
                    min_ver_dist = min(self.y_max - point[1], point[1] - self.y_min)
                    max_dist = np.hypot(min_hor_dist, min_ver_dist)

                    # w = np.ones_like(distances)
                    mask = distances <= max_dist
                    w1 = (
                        1
                        - (
                            np.arccos(np.minimum(min_ver_dist, distances) / distances *mask)
                            + np.arccos(np.minimum(min_hor_dist, distances) / distances *mask)
                        )
                        / np.pi
                    )

                    w2 = (
                        3/4
                        - 0.5
                        * (
                            np.arccos(min_ver_dist / distances * ~mask)
                            + np.arccos(min_hor_dist / distances * ~mask)
                        )
                        / np.pi
                    )

                    w = mask * w1 + ~mask * w2

                    bivariate_ripley[r] += np.sum((distances < radii[r]) / w)

            bivariate_ripley = self.area * bivariate_ripley / (npts1 * npts2)

        elif mode == "extend":
            for r in range(len(radii)):
                for point in data1:
                    # distances = np.sqrt((ext_data2[:, 0] - point[0]) ** 2 + (ext_data2[:, 1] - point[1]) ** 2)
                    distances = np.hypot((ext_data2[:, 0] - point[0]), (ext_data2[:, 1] - point[1]))
                    bivariate_ripley[r] += (distances < radii[r]).sum()
            bivariate_ripley = self.area * bivariate_ripley / (npts1 * npts2)
        elif mode == "merge_ext":
            for r in range(len(radii)):
                for point in data1:
                    # distances = np.sqrt((ext_data2[:, 0] - point[0]) ** 2 + (ext_data2[:, 1] - point[1]) ** 2)
                    distances = np.hypot((ext_data2[:, 0] - point[0]), (ext_data2[:, 1] - point[1]))
                    bivariate_ripley[r] += (distances < radii[r]).sum()
            bivariate_ripley = self.area * bivariate_ripley
     
        return bivariate_ripley

    # Method 5
    def evaluate_weighted_bivariate_extendMod(self, data1, data2, ext_data1, ext_data2, radii, mode="extend"):
        npts1 = len(data1)
        npts2 = len(data2)

        lambda1 = npts1/self.area
        lambda2 = npts2/self.area

        return (lambda2 * self.evaluate_bivariate_extendMod(data1, ext_data2, npts2, radii, mode=mode) + 
                lambda1 * self.evaluate_bivariate_extendMod(data2, ext_data1, npts1, radii, mode=mode)) / (lambda1 + lambda2)
    
    # Utility Internal function
    def divide_data_extend(self, ext_data, boundaries, sub_size, n_sections, c_search):
        """
        c_search: defines how much more area we use to count astrocytes (offspring cells) for the neurons (parent cells)
                    for example, if we calculate the ripley's k value for an 900 x 900 area, and c_search=0.1, then we take 
                    into account the astrocytes within the 990 x 990 area, while taking into account the neurons within the 900 x 900 area.
        n_sections: define the grid size we divide into. for example, for 900 x 900 entire area, if n_section=3, then we calculate the
                    ripley's k for 300 x 300 areas and divide the 900x900 area to nine 300x300 areas. (9=3x3, 300=900/3)
        sub_size: the length of edge after division. For example, sub_size=300 if we divide it into 300x300.
        
        sub_size x n_sections = edge length
          300    x      3     = 900             for example.

        if sub_size can't be divided evenly by the edge length, for example, sub_size = 500 when we calculate for a 900x900 area,
        then the function use n_sections as primary parameter and discard the sub_size parameter.
        """


        list_of_sub_ext_data = [[] for _ in range(len(boundaries))]
        
        for k,boundary in enumerate(boundaries):
            x_0, x_1, y_0, y_1 = boundary


            for i,row in enumerate(ext_data):
                x = row[0]
                y = row[1]

                if x >= x_0 - c_search * sub_size and \
                    x <= x_1 + c_search * sub_size and \
                    y >= y_0 - c_search * sub_size and \
                    y <= y_1 + c_search * sub_size:
                    
                    list_of_sub_ext_data[k].append(row)

        divided_ext_data = [np.array(sublist) for sublist in list_of_sub_ext_data]

        return divided_ext_data

    # Method 7
    def divide_n_conquer_extend(self, data1, data2, ext_data2, radii, mode="extend", sub_size=100, n_sections=7, c_search=0.1):
        """
        c_search: defines how much more area we use to count astrocytes (offspring cells) for the neurons (parent cells)
                    for example, if we calculate the ripley's k value for an 900 x 900 area, and c_search=0.1, then we take 
                    into account the astrocytes within the 990 x 990 area, while taking into account the neurons within the 900 x 900 area.
        n_sections: define the grid size we divide into. for example, for 900 x 900 entire area, if n_section=3, then we calculate the
                    ripley's k for 300 x 300 areas and divide the 900x900 area to nine 300x300 areas. (9=3x3, 300=900/3)
        sub_size: the length of edge after division. For example, sub_size=300 if we divide it into 300x300.
        
        sub_size x n_sections = edge length
          300    x      3     = 900             for example.

        if sub_size can't be divided evenly by the edge length, for example, sub_size = 500 when we calculate for a 900x900 area,
        then the function use n_sections as primary parameter and discard the sub_size parameter.
        """

        boundaries, sub_size, n_sections = self.divide_boundaries(sub_size, n_sections)
        
        divided_data1 = self.divide_data(data1, boundaries, sub_size, n_sections)
        
        # # simple visualize
        # # print(extended_B)
        # plt.scatter(divided_data1[5][:, 0], divided_data1[5][:, 1])
        # plt.title('Scatter Plot of unExtended Points')
        # plt.xlabel('X axis')
        # plt.ylabel('Y axis')
        # plt.grid(True)
        # plt.show()

        
        divided_data2 = self.divide_data(data2, boundaries, sub_size, n_sections)
        # # simple visualize
        # # print(extended_B)
        # plt.scatter(divided_data2[5][:, 0], divided_data2[5][:, 1])
        # plt.title('Scatter Plot of unExtended Points')
        # plt.xlabel('X axis')
        # plt.ylabel('Y axis')
        # plt.grid(True)
        # plt.show()

        
        
        ext_divided_data2  = self.divide_data_extend(ext_data2, boundaries, sub_size, n_sections, c_search)

        # plt.scatter(ext_divided_data2[5][:, 0], ext_divided_data2[5][:, 1])
        # plt.title('Scatter Plot of Extended Points')
        # plt.xlabel('X axis')
        # plt.ylabel('Y axis')
        # plt.grid(True)
        # plt.show()

        npts2_all = sum([len(i) for i in ext_divided_data2])
        # print("npts2_all=",npts2_all)

        results = []

        # return
        for i,boundary in enumerate(boundaries):
            self.x_min, self.x_max, self.y_min, self.y_max = boundary
            if len(ext_divided_data2[i]) > 1:
                npts2 = len(divided_data2[i])
                # self.area = self.area / (n_sections * n_sections)
                result = self.evaluate_bivariate_extendMod(divided_data1[i], ext_divided_data2[i], npts2, radii, mode="merge_ext")
                # self.area = self.area * (n_sections * n_sections)
                # npts2_ext_divided = len(ext_divided_data2[i])

                # npts1_divided = len(divided_data1)
                # npts2_divided = len(divided_data2)
                # weight = npts1_divided * npts2_divided
                # results.append(weight * result)
                
                results.append(result)
                # print("npts2=",len(ext_divided_data2[i]))
            else:
                result = np.zeros(len(radii))
                results.append(result)
        
        results = np.asarray(results)
        
        # average_result = results.mean(axis=0) * len(boundaries) / npts2_all / 12.5
        # average_result = results.sum(axis=0) / npts2_all  * 2
        # average_result = results.sum(axis=0) / len(data1) / len(data2) * 12
        npts1 = len(data1)
        npts2 = len(data2)
        average_result = results.sum(axis=0) / npts1 / npts2

        self.x_min, self.x_max, self.y_min, self.y_max = self.origional_boundary
        return average_result
    
#    Method 8
    def evaluate_bivariate_cell_wise(self, point1, data2, radii, mode="none"):
        """
        Evaluates the bivariate Ripley's K function for each neurons.

        Parameters
        ----------
        point1: 1D tuple
                (x_pixel, y_pixel) of a Neuron position.
        An external loop will iterate through all neurons.

        data2 : 2D array 
            Nx3 array,
            where each row is an astrocyte, (x_pixel, y_pixel, gene_expression)
                                            where gene_expression is the value in the X matrix for a selected biomarker gene.
            Set of observed points of type 2 (astrocytes positions+gene expression).

        radii : 1D array
            Set of distances at which the bivariate K function will be evaluated.
        mode : str
            Mode for edge effect correction. Options is 'none'; "Ripley" does not apply for this method.
            An external process is done for edge correction under cell-wise calculation, requiring TESLA package.

        Extra Author comment: the result is similar with/without edge correction.

        Returns
        -------
        bivariate_ripley : 1D array
            Bivariate Ripley's K function evaluated at `radii`.
        """
        # print("evalulating, mode:", mode)
        if mode not in ["none"]:
            raise ValueError(f"mode {mode} is not implemented.")

        data2 = np.asarray(data2)
        npts2 = len(data2)
        bivariate_ripley = np.zeros(len(radii))

        if mode == "ripley":
            mode = "none"
            print( "'Ripley' does not apply for this method. \n An external process is done for edge correction under cell-wise calculation, requiring TESLA package.")
            print( "'none' mode is calculated.")
            print(" Extra comment: the result is similar with/without edge correction.")

        if mode == "none":
            for r in range(len(radii)):                
                distances = np.hypot((data2[:, 0] - point1[0]), (data2[:, 1] - point1[1]))
                bivariate_ripley[r] += (distances < radii[r]).sum()
            # bivariate_ripley = self.area * bivariate_ripley / (npts2)
        
        return bivariate_ripley

    def mark_weighted_evaluate_bivariate_cell_wise(self, point1, data2, radii, mode="none"):
        """
        Evaluates the bivariate Ripley's K function for each neurons.

        Parameters
        ----------
        point1: 1D tuple
                (x_pixel, y_pixel) of a Neuron position.
        An external loop will iterate through all neurons.

        data2 : 2D array 
            Nx3 array,
            where each row is an astrocyte, (x_pixel, y_pixel, gene_expression)
                                            where gene_expression is the value in the X matrix for a selected biomarker gene.
            Set of observed points of type 2 (astrocytes positions+gene expression).

        radii : 1D array
            Set of distances at which the bivariate K function will be evaluated.
        mode : str
            Mode for edge effect correction. Options is 'none'; "Ripley" does not apply for this method.
            An external process is done for edge correction under cell-wise calculation, requiring TESLA package.

        Extra Author comment: the result is similar with/without edge correction.

        Returns
        -------
        bivariate_ripley : 1D array
            Bivariate Ripley's K function evaluated at `radii`.
        """

        # print("evalulating, mode:", mode)
        if mode not in ["none"]:
            raise ValueError(f"mode {mode} is not implemented.")
        if mode == "ripley":
            mode = "none"
            print( "'Ripley' does not apply for this method. \n An external process is done for edge correction under cell-wise calculation, requiring TESLA package.")
            print( "'none' mode is calculated.")
            print(" Extra comment: the result is similar with/without edge correction.")

        data2 = np.asarray(data2)
        npts2 = len(data2)
        bivariate_ripley = np.zeros(len(radii))

        if mode == "none":
            for r in range(len(radii)):                
                distances = np.hypot((data2[:, 0] - point1[0]), (data2[:, 1] - point1[1]))
                
                within_radius = distances < radii[r]
                # Transfer the Bool Index into integers, then multiply by gene expression and do a weighted sum
                bivariate_ripley[r] += (within_radius.astype(int) * data2[:, 2]).sum()

            # bivariate_ripley = self.area * bivariate_ripley / (npts2)
        
        return bivariate_ripley

    def cupy_mark_weighted_evaluate_bivariate_cell_wise(self, neurons_batch, astrocytes, radii, mode="none"):
        """
        Evaluates the bivariate Ripley's K function for each neurons.

        Parameters
        ----------
        point1: 1D tuple
                (x_pixel, y_pixel) of a Neuron position.
        An external loop will iterate through all neurons.

        data2 : 2D array 
            Nx3 array,
            where each row is an astrocyte, (x_pixel, y_pixel, gene_expression)
                                            where gene_expression is the value in the X matrix for a selected biomarker gene.
            Set of observed points of type 2 (astrocytes positions+gene expression).

        radii : 1D array
            Set of distances at which the bivariate K function will be evaluated.
        mode : str
            Mode for edge effect correction. Options is 'none'; "Ripley" does not apply for this method.
            An external process is done for edge correction under cell-wise calculation, requiring TESLA package.

        Extra Author comment: the result is similar with/without edge correction.

        Returns
        -------
        bivariate_ripley : 1D array
            Bivariate Ripley's K function evaluated at `radii`.
        """

        # print("evalulating, mode:", mode)
        if mode not in ["none"]:
            raise ValueError(f"mode {mode} is not implemented.")
        if mode == "ripley":
            mode = "none"
            print( "'Ripley' does not apply for this method. \n An external process is done for edge correction under cell-wise calculation, requiring TESLA package.")
            print( "'none' mode is calculated.")
            print(" Extra comment: the result is similar with/without edge correction.")
            
        # 数据传输到 GPU
        neurons_batch_gpu = cp.asarray(neurons_batch)  # 神经元块
        astrocytes_gpu = cp.asarray(astrocytes)        # 星形胶质细胞数据
        ripley_values_gpu = cp.zeros((neurons_batch_gpu.shape[0], len(radii)))  # 初始化结果

        # 逐个半径计算 Ripley's K
        for i, r in enumerate(radii):
            # 计算神经元到所有星形胶质细胞的距离
            distances = cp.hypot(
                neurons_batch_gpu[:, 0][:, None] - astrocytes_gpu[:, 0][None, :],
                neurons_batch_gpu[:, 1][:, None] - astrocytes_gpu[:, 1][None, :]
            )
            # 找到距离在半径 r 内的星形胶质细胞
            within_radius = (distances < r).astype(cp.float32)
            # print(within_radius)
            # print(within_radius.shape)
            # print(astrocytes_gpu[:,2])
            # print(astrocytes_gpu[:,2].shape)
            # 累积基因表达值
            ripley_values_gpu[:, i] = cp.sum(
                within_radius * astrocytes_gpu[:, 2][None, :], axis=1
            )
        
        # 将结果传回 CPU
        ripley_values = cp.asnumpy(ripley_values_gpu)
        return ripley_values


    def cupy2_mark_weighted_evaluate_bivariate_cell_wise(self, neurons_batch, astrocytes, radii, mode="none"):
        """
        Evaluates the bivariate Ripley's K function for each neurons.

        Parameters
        ----------
        point1: 1D tuple
                (x_pixel, y_pixel) of a Neuron position.
        An external loop will iterate through all neurons.

        data2 : 2D array 
            Nx3 array,
            where each row is an astrocyte, (x_pixel, y_pixel, gene_expression)
                                            where gene_expression is the value in the X matrix for a selected biomarker gene.
            Set of observed points of type 2 (astrocytes positions+gene expression).

        radii : 1D array
            Set of distances at which the bivariate K function will be evaluated.
        mode : str
            Mode for edge effect correction. Options is 'none'; "Ripley" does not apply for this method.
            An external process is done for edge correction under cell-wise calculation, requiring TESLA package.

        Extra Author comment: the result is similar with/without edge correction.

        Returns
        -------
        bivariate_ripley : 1D array
            Bivariate Ripley's K function evaluated at `radii`.
        """

        # print("evalulating, mode:", mode)
        if mode not in ["none"]:
            raise ValueError(f"mode {mode} is not implemented.")
        if mode == "ripley":
            mode = "none"
            print( "'Ripley' does not apply for this method. \n An external process is done for edge correction under cell-wise calculation, requiring TESLA package.")
            print( "'none' mode is calculated.")
            print(" Extra comment: the result is similar with/without edge correction.")
            
        neurons_batch_gpu = neurons_batch
        astrocytes_gpu = astrocytes
        ripley_values_gpu = np.zeros((neurons_batch_gpu.shape[0], len(radii)))  # 初始化结果

        # 逐个半径计算 Ripley's K
        for i, r in enumerate(radii):
            # 计算神经元到所有星形胶质细胞的距离
            distances = np.hypot(
                neurons_batch_gpu[:, 0][:, None] - astrocytes_gpu[:, 0][None, :],
                neurons_batch_gpu[:, 1][:, None] - astrocytes_gpu[:, 1][None, :]
            )
            # 找到距离在半径 r 内的星形胶质细胞
            within_radius = (distances < r).astype(np.float32)
            # print(within_radius)
            # print(within_radius.shape)
            # print(astrocytes_gpu[:,2])
            # print(astrocytes_gpu[:,2].shape)
            # 累积基因表达值
            ripley_values_gpu[:, i] = np.sum(
                within_radius * astrocytes_gpu[:, 2][None, :], axis=1
            )
        
        # 将结果传回 CPU
        ripley_values = (ripley_values_gpu)
        return ripley_values


    def cupy3_mark_weighted_evaluate_bivariate_cell_wise(self, 
                                                         neurons_batch, 
                                                         astrocytes, 
                                                         radii, 
                                                         mode="none"):
        """
        新增函数：与 cupy_mark_weighted_evaluate_bivariate_cell_wise 相同计算逻辑，
        但会对每个关键步骤进行计时并打印，用以分析性能瓶颈。
        """

        if mode not in ["none"]:
            raise ValueError(f"mode {mode} is not implemented.")
        if mode == "ripley":
            mode = "none"
            print("'Ripley' does not apply for this method.\n"
                  "An external process is done for edge correction under cell-wise calculation, requiring TESLA package.\n"
                  "'none' mode is calculated.\n"
                  "Extra comment: the result is similar with/without edge correction.")

        # ====== 1) 将数据拷贝到 GPU 上 ======
        t0 = time.time()
        cp.cuda.stream.get_current_stream().synchronize()

        neurons_batch_gpu = cp.asarray(neurons_batch)  # 神经元块
        t1 = time.time()
        cp.cuda.stream.get_current_stream().synchronize()
        print(f"[cupy3] Time for transferring neurons_batch to GPU: {t1 - t0:.4f} s")

        astrocytes_gpu = cp.asarray(astrocytes)        # 星形胶质细胞数据
        t2 = time.time()
        cp.cuda.stream.get_current_stream().synchronize()
        print(f"[cupy3] Time for transferring astrocytes to GPU: {t2 - t1:.4f} s")

        # ====== 2) 分配结果数组 ======
        ripley_values_gpu = cp.zeros((neurons_batch_gpu.shape[0], len(radii)))
        t3 = time.time()
        cp.cuda.stream.get_current_stream().synchronize()
        print(f"[cupy3] Time for allocating ripley_values_gpu: {t3 - t2:.4f} s")

        # ====== 3) 逐个半径计算 Ripley's K ======
        loop_start = time.time()
        cp.cuda.stream.get_current_stream().synchronize()

        for i, r in enumerate(radii):
            # -- 3.1) 计算距离矩阵 --
            dist_start = time.time()
            distances = cp.hypot(
                neurons_batch_gpu[:, 0][:, None] - astrocytes_gpu[:, 0][None, :],
                neurons_batch_gpu[:, 1][:, None] - astrocytes_gpu[:, 1][None, :]
            )
            cp.cuda.stream.get_current_stream().synchronize()
            dist_end = time.time()

            # -- 3.2) 找到距离在 r 内的星形胶质细胞 & 累加表达 --
            within_start = time.time()
            within_radius = (distances < r).astype(cp.float32)
            cp.cuda.stream.get_current_stream().synchronize()
            within_end = time.time()

            sum_start = time.time()
            ripley_values_gpu[:, i] = cp.sum(
                within_radius * astrocytes_gpu[:, 2][None, :], axis=1
            )
            cp.cuda.stream.get_current_stream().synchronize()
            sum_end = time.time()

            print(f"[cupy3] Radius index {i}, distance calc: {dist_end - dist_start:.4f}s, "
                  f"within+astype: {within_end - within_start:.4f}s, "
                  f"sum: {sum_end - sum_start:.4f}s")

        loop_end = time.time()
        print(f"[cupy3] >>> For-loop over all radii took {loop_end - loop_start:.4f} s in total.")

        # ====== 4) 将结果传回 CPU ======
        t4 = time.time()
        cp.cuda.stream.get_current_stream().synchronize()

        ripley_values = cp.asnumpy(ripley_values_gpu)
        t5 = time.time()
        cp.cuda.stream.get_current_stream().synchronize()
        print(f"[cupy3] Time for transferring ripley_values back to CPU: {t5 - t4:.4f} s")

        # ====== 总时间 ======
        print(f"[cupy3] TOTAL function time (excluding prints) ~ {t5 - t0:.4f} s\n")

        return ripley_values

    # --------------------------------------------------------------------------
    # 新函数：一次性计算距离，且全程用 float32，并跟 cupy3 一样输出计时信息
    # --------------------------------------------------------------------------
    def cupy4_mark_weighted_evaluate_bivariate_cell_wise(self, 
                                                         neurons_batch, 
                                                         astrocytes, 
                                                         radii, 
                                                         mode="none"):
        """
        1. 将 neurons_batch, astrocytes (x, y, gene_expr) 都转成 float32
        2. 只计算一次距离矩阵 distances，然后对每个半径做 (distances < r) 比较
        3. 用计时方法打印各步骤耗时
        """
        if mode not in ["none"]:
            raise ValueError(f"mode {mode} is not implemented.")
        if mode == "ripley":
            mode = "none"
            print("'Ripley' does not apply for this method.\n"
                  "An external process is done for edge correction under cell-wise calculation, requiring TESLA package.\n"
                  "'none' mode is calculated.\n"
                  "Extra comment: the result is similar with/without edge correction.")

        # ====== 0) 计时器起点 ======
        t0 = time.time()
        cp.cuda.stream.get_current_stream().synchronize()

        # ====== 1) 数据拷贝到 GPU 并使用 float32 ======
        # 注意：坐标及基因表达数据都转 float32
        neurons_batch_gpu = cp.asarray(neurons_batch, dtype=cp.float32)
        astrocytes_gpu    = cp.asarray(astrocytes,    dtype=cp.float32)

        t1 = time.time()
        cp.cuda.stream.get_current_stream().synchronize()
        print(f"[cupy4] Time for transferring data to GPU (float32): {t1 - t0:.4f} s")

        # ====== 2) 先一次性分配好返回结果 ======
        ripley_values_gpu = cp.zeros((neurons_batch_gpu.shape[0], len(radii)), dtype=cp.float32)

        t2 = time.time()
        cp.cuda.stream.get_current_stream().synchronize()
        print(f"[cupy4] Time for allocating ripley_values_gpu: {t2 - t1:.4f} s")

        # ====== 3) 只计算一次距离矩阵 ======
        dist_start = time.time()

        distances = cp.hypot(
            neurons_batch_gpu[:, 0][:, None] - astrocytes_gpu[:, 0][None, :],
            neurons_batch_gpu[:, 1][:, None] - astrocytes_gpu[:, 1][None, :]
        )

        cp.cuda.stream.get_current_stream().synchronize()
        dist_end = time.time()
        print(f"[cupy4] Time for computing full distances matrix: {dist_end - dist_start:.4f} s")

        # ====== 4) 对每个半径做比较并 sum ======
        loop_start = time.time()

        expr_gpu = astrocytes_gpu[:, 2]  # 基因表达列
        for i, r in enumerate(radii):
            step_start = time.time()

            within_radius = (distances < r).astype(cp.float32) 
            cp.cuda.stream.get_current_stream().synchronize()
            step_mid = time.time()

            ripley_values_gpu[:, i] = cp.sum(within_radius * expr_gpu[None, :], axis=1)
            cp.cuda.stream.get_current_stream().synchronize()
            step_end = time.time()

            print(f"[cupy4] Radius index {i}, compare: {step_mid - step_start:.4f}s, sum: {step_end - step_mid:.4f}s")

        loop_end = time.time()
        print(f"[cupy4] >>> For-loop over all radii took {loop_end - loop_start:.4f} s in total.")

        # ====== 5) 结果拷回 CPU ======
        t3 = time.time()
        cp.cuda.stream.get_current_stream().synchronize()

        ripley_values = cp.asnumpy(ripley_values_gpu)  # shape: (batch_size, len(radii))

        t4 = time.time()
        cp.cuda.stream.get_current_stream().synchronize()
        print(f"[cupy4] Time for transferring result back to CPU: {t4 - t3:.4f} s")

        # ====== 6) 最终耗时 ======
        print(f"[cupy4] TOTAL function time ~ {t4 - t0:.4f} s\n")

        return ripley_values


    # --------------------------------------------------------------------------
    # 新函数：一次性计算距离，且全程用 float32，不输出计时信息
    # --------------------------------------------------------------------------
    def cupy5_mark_weighted_evaluate_bivariate_cell_wise(self, 
                                                         neurons_batch, 
                                                         astrocytes, 
                                                         radii, 
                                                         mode="none"):
        """
        1. 将 neurons_batch, astrocytes (x, y, gene_expr) 都转成 float32
        2. 只计算一次距离矩阵 distances，然后对每个半径做 (distances < r) 比较
        3. 用计时方法打印各步骤耗时
        """
        if mode not in ["none"]:
            raise ValueError(f"mode {mode} is not implemented.")
        if mode == "ripley":
            mode = "none"
            print("'Ripley' does not apply for this method.\n"
                  "An external process is done for edge correction under cell-wise calculation, requiring TESLA package.\n"
                  "'none' mode is calculated.\n"
                  "Extra comment: the result is similar with/without edge correction.")

        # ====== 0) 计时器起点 ======
        cp.cuda.stream.get_current_stream().synchronize()

        # ====== 1) 数据拷贝到 GPU 并使用 float32 ======
        # 注意：坐标及基因表达数据都转 float32
        neurons_batch_gpu = cp.asarray(neurons_batch, dtype=cp.float32)
        astrocytes_gpu    = cp.asarray(astrocytes,    dtype=cp.float32)

        cp.cuda.stream.get_current_stream().synchronize()

        # ====== 2) 先一次性分配好返回结果 ======
        ripley_values_gpu = cp.zeros((neurons_batch_gpu.shape[0], len(radii)), dtype=cp.float32)

        cp.cuda.stream.get_current_stream().synchronize()

        # ====== 3) 只计算一次距离矩阵 ======
        dist_start = time.time()

        distances = cp.hypot(
            neurons_batch_gpu[:, 0][:, None] - astrocytes_gpu[:, 0][None, :],
            neurons_batch_gpu[:, 1][:, None] - astrocytes_gpu[:, 1][None, :]
        )

        cp.cuda.stream.get_current_stream().synchronize()

        # ====== 4) 对每个半径做比较并 sum ======

        expr_gpu = astrocytes_gpu[:, 2]  # 基因表达列
        for i, r in enumerate(radii):

            within_radius = (distances < r).astype(cp.float32) 
            cp.cuda.stream.get_current_stream().synchronize()

            ripley_values_gpu[:, i] = cp.sum(within_radius * expr_gpu[None, :], axis=1)
            cp.cuda.stream.get_current_stream().synchronize()



        # ====== 5) 结果拷回 CPU ======
        cp.cuda.stream.get_current_stream().synchronize()

        ripley_values = cp.asnumpy(ripley_values_gpu)  # shape: (batch_size, len(radii))

        cp.cuda.stream.get_current_stream().synchronize()

        # ====== 6) 最终耗时 ======

        return ripley_values


    # --------------------------------------------------------------------------
    # 新函数：一次性计算距离，且全程用 float32，不输出计时信息，astrocytes (Gcells)预先传入GPU
    # --------------------------------------------------------------------------
    def cupy6_mark_weighted_evaluate_bivariate_cell_wise(self, 
                                                         neurons_batch, 
                                                         astrocytes, 
                                                         radii, 
                                                         mode="none"):
        """
        1. 将 neurons_batch, astrocytes (x, y, gene_expr) 都转成 float32
        2. 只计算一次距离矩阵 distances，然后对每个半径做 (distances < r) 比较
        3. 用计时方法打印各步骤耗时
        """
        if mode not in ["none"]:
            raise ValueError(f"mode {mode} is not implemented.")
        if mode == "ripley":
            mode = "none"
            print("'Ripley' does not apply for this method.\n"
                  "An external process is done for edge correction under cell-wise calculation, requiring TESLA package.\n"
                  "'none' mode is calculated.\n"
                  "Extra comment: the result is similar with/without edge correction.")

        # ====== 0) 计时器起点 ======
        cp.cuda.stream.get_current_stream().synchronize()

        # ====== 1) 数据拷贝到 GPU 并使用 float32 ======
        # 注意：坐标及基因表达数据都转 float32
        neurons_batch_gpu = cp.asarray(neurons_batch, dtype=cp.float32)
        # astrocytes_gpu    = cp.asarray(astrocytes,    dtype=cp.float32)
        astrocytes_gpu    = astrocytes

        cp.cuda.stream.get_current_stream().synchronize()

        # ====== 2) 先一次性分配好返回结果 ======
        ripley_values_gpu = cp.zeros((neurons_batch_gpu.shape[0], len(radii)), dtype=cp.float32)

        cp.cuda.stream.get_current_stream().synchronize()

        # ====== 3) 只计算一次距离矩阵 ======
        dist_start = time.time()

        distances = cp.hypot(
            neurons_batch_gpu[:, 0][:, None] - astrocytes_gpu[:, 0][None, :],
            neurons_batch_gpu[:, 1][:, None] - astrocytes_gpu[:, 1][None, :]
        )

        cp.cuda.stream.get_current_stream().synchronize()

        # ====== 4) 对每个半径做比较并 sum ======

        expr_gpu = astrocytes_gpu[:, 2]  # 基因表达列
        for i, r in enumerate(radii):

            within_radius = (distances < r).astype(cp.float32) 
            cp.cuda.stream.get_current_stream().synchronize()

            ripley_values_gpu[:, i] = cp.sum(within_radius * expr_gpu[None, :], axis=1)
            cp.cuda.stream.get_current_stream().synchronize()



        # ====== 5) 结果拷回 CPU ======
        cp.cuda.stream.get_current_stream().synchronize()

        ripley_values = cp.asnumpy(ripley_values_gpu)  # shape: (batch_size, len(radii))

        cp.cuda.stream.get_current_stream().synchronize()

        # ====== 6) 最终耗时 ======

        return ripley_values


class RipleysK3(ripleyk):
    pass