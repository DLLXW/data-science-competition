class EstimatorCV():
    def __init__(self, feature_num, class_num, device):
        super(EstimatorCV, self).__init__()

        self.class_num = class_num
        self.device = device
        self.CoVariance = torch.zeros(class_num, feature_num).to(device)
        self.Ave = torch.zeros(class_num, feature_num).to(device)
        self.Amount = torch.zeros(class_num).to(device)

    def update_CV(self, features, labels):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        NxCxFeatures = features.view(
            N, 1, A
        ).expand(
            N, C, A
        )

        label_mask = (labels == 255).long()
        labels = ((1 - label_mask).mul(labels) + label_mask * 19).long()

        onehot = torch.zeros(N, C).cuda()
        onehot.scatter_(1, labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - \
                   ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

        var_temp = var_temp.pow(2).sum(0).div(Amount_CxA)

        sum_weight_CV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(
            sum_weight_CV + self.Amount.view(C, 1).expand(C, A)
        )

        weight_CV[weight_CV != weight_CV] = 0

        additional_CV = weight_CV.mul(1 - weight_CV).mul((self.Ave - ave_CxA).pow(2))

        self.CoVariance = (self.CoVariance.mul(1 - weight_CV) + var_temp
                           .mul(weight_CV)).detach() + additional_CV.detach()

        self.Ave = (self.Ave.mul(1 - weight_CV) + ave_CxA.mul(weight_CV)).detach()

        self.Amount += onehot.sum(0)


class ISDALoss(nn.Module):
    def __init__(self, feature_num, class_num, device):
        super(ISDALoss, self).__init__()

        self.estimator = EstimatorCV(feature_num, class_num + 1, device)

        self.class_num = class_num

        self.cross_entropy = nn.CrossEntropyLoss()

    def isda_aug(self, fc, features, y, labels, cv_matrix, ratio):
        label_mask = (labels == 255).long()
        labels = (1 - label_mask).mul(labels).long()

        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        weight_m = list(fc.parameters())[0].squeeze()

        NxW_ij = weight_m.expand(N, C, A)

        NxW_kj = torch.gather(NxW_ij,
                              1,
                              labels.view(N, 1, 1)
                              .expand(N, C, A))

        CV_temp = cv_matrix[labels]

        sigma2 = ratio * (weight_m - NxW_kj).pow(2).mul(
            CV_temp.view(N, 1, A).expand(N, C, A)
        ).sum(2)

        aug_result = y + 0.5 * sigma2.mul((1 - label_mask).view(N, 1).expand(N, C))

        return aug_result

    def forward(self, features, final_conv, y, target_x, ratio):
        # features = model(x)

        N, A, H, W = features.size()

        target_x = target_x.view(N, 1, target_x.size(1), target_x.size(2)).float()

        target_x = F.interpolate(target_x, size=(H, W), mode='nearest', align_corners=None)

        target_x = target_x.long().squeeze()

        C = self.class_num

        features_NHWxA = features.permute(0, 2, 3, 1).contiguous().view(N * H * W, A)

        target_x_NHW = target_x.contiguous().view(N * H * W)

        y_NHWxC = y.permute(0, 2, 3, 1).contiguous().view(N * H * W, C)

        self.estimator.update_CV(features_NHWxA.detach(), target_x_NHW)

        isda_aug_y_NHWxC = self.isda_aug(final_conv, features_NHWxA, y_NHWxC, target_x_NHW,
                                         self.estimator.CoVariance.detach(), ratio)

        isda_aug_y = isda_aug_y_NHWxC.view(N, H, W, C).permute(0, 3, 1, 2)

        return isda_aug_y