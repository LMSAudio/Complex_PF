from git_config import *
from model.istft import ISTFT
import torch.optim.lr_scheduler as lrSched

# Complex CNN for online processing.
# Written for: Microsoft AEC Challenge 2021
# Modar Halimeh. LMS. 2020.
# Modified after: Phase-ware speech enhancement and Deep Complex U-net


device = 'cuda:0'

n_fft       = 424
hop_length  = 212
window = torch.hann_window(n_fft).to(device)
stft = lambda x: torch.stft(x, n_fft, hop_length, window=window,  center=True)
istft = ISTFT(n_fft, hop_length, window='hanning').to(device)


def pad2d_as(x1, x2):
    # Pad x1 to have same size with x2
    # inputs are NCHW
    diffH = x2.size()[2] - x1.size()[2]
    diffW = x2.size()[3] - x1.size()[3]

    return F.pad(x1, (0, diffW, 0, diffH))

def padded_cat(x1, x2, dim):
    x1 = pad2d_as(x1, x2)
    x1 = torch.cat([x1, x2], dim=dim)
    return x1

class Encoder(nn.Module): #describes one encoder layer
    def __init__(self, conv_cfg, leaky_slope):
        super(Encoder, self).__init__()
        self.conv = complexnn.ComplexConvWrapper(nn.Conv2d, *conv_cfg, bias=False)
        self.bn = complexnn.ComplexBatchNorm(conv_cfg[1])
        self.act = complexnn.CLeakyReLU(leaky_slope, inplace=True)

    def forward(self, xr, xi):
        xr, xi = self.act(*self.bn(*self.conv(xr, xi)))
        return xr, xi

class Decoder(nn.Module): #describes one decoder layer
    def __init__(self, dconv_cfg, leaky_slope):
        super(Decoder, self).__init__()
        self.dconv = complexnn.ComplexConvWrapper(nn.ConvTranspose2d, *dconv_cfg, bias=False)
        self.bn = complexnn.ComplexBatchNorm(dconv_cfg[1])
        self.act = complexnn.CLeakyReLU(leaky_slope, inplace=True)

    def forward(self, xr, xi, skip=None):
        if skip is not None:
            xr, xi = padded_cat(xr, skip[0], dim=1), padded_cat(xi, skip[1], dim=1)
        xr, xi = self.act(*self.bn(*self.dconv(xr, xi)))
        return xr, xi

class ComplexGRU(nn.Module):
    def __init__(self, inputSize= 64*17*2, hiddenSize= 128, numLayers =1, batchSize=1):
        super(ComplexGRU, self).__init__()

        self.hiddenSize = hiddenSize

        self.grulayerR = torch.nn.GRU(input_size=inputSize, hidden_size=hiddenSize, num_layers=numLayers, bias=False, batch_first=True)
        self.grulayerI = torch.nn.GRU(input_size=inputSize, hidden_size=hiddenSize, num_layers=numLayers, bias=False, batch_first=True)

        self.FCLr      = torch.nn.Linear(in_features=hiddenSize, out_features=inputSize, bias=False)
        self.FCLi      = torch.nn.Linear(in_features=hiddenSize, out_features=inputSize, bias=False)
        self.act       = nn.LeakyReLU(0.1)

        self.register_buffer('hrrOInit', torch.zeros(1, batchSize, hiddenSize))

        self.hrrO, self.hiiO, self.hirO, self.hriO =  self.hrrOInit[:, 0:batchSize,:], self.hrrOInit[:, 0:batchSize,:], self.hrrOInit[:, 0:batchSize,:], self.hrrOInit[:, 0:batchSize,:]

    def reset_internal_stat(self, batchSize = 1, Device=device):

        self.register_buffer('hrrOInit', torch.zeros(1, batchSize, self.hiddenSize))

        if Device.type=='cuda':
            self.hrrOInit = self.hrrOInit.cuda()

        self.hrrO, self.hiiO, self.hirO, self.hriO =  self.hrrOInit[:, 0:batchSize,:], self.hrrOInit[:, 0:batchSize,:], self.hrrOInit[:, 0:batchSize,:], self.hrrOInit[:, 0:batchSize,:]


    def forward(self, xr, xi):

        originalShape = xr.shape

        xr = torch.reshape(xr, [-1, 1, originalShape[3]*originalShape[2]*originalShape[1]])
        xi = torch.reshape(xi, [-1, 1, originalShape[3]*originalShape[2]*originalShape[1]])

        GRU_rr, self.hrrO = self.grulayerR(xr, self.hrrO)
        GRU_ri, self.hriO = self.grulayerI(xr, self.hriO)

        GRU_ii, self.hiiO = self.grulayerI(xi, self.hiiO)
        GRU_ir, self.hirO = self.grulayerR(xi, self.hirO)


        xr_gru = GRU_rr - GRU_ii
        xi_gru = GRU_ir + GRU_ri

        xr = self.FCLr(xr_gru) - self.FCLi(xi_gru)
        xi = self.FCLi(xr_gru) + self.FCLr(xi_gru)

        xr, xi = torch.reshape(xr, [originalShape[0], originalShape[1], originalShape[2],
                                          originalShape[3]]), torch.reshape(xi, [originalShape[0], originalShape[1], originalShape[2], originalShape[3]])
        return xr, xi


class complexPF(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.encoders = nn.ModuleList()
        for layer in range(0, 4):
            self.encoders.append(Encoder(cfg['encoders'][layer], 0.1))

        self.decoders = nn.ModuleList()
        for layer in range(0, 3):
            self.decoders.append(Decoder(cfg['decoders'][layer], 0.1))

        self.final_dec = complexnn.ComplexConvWrapper(nn.ConvTranspose2d, *cfg['decoders'][3], bias=True)

        self.gruLayer = ComplexGRU(inputSize=cfg['GRUdim'], hiddenSize=cfg['GRUdim'], numLayers=1, batchSize=1)

        self.hop_length         = cfg['hop_length']
        self.nrFrames           = cfg['nrFrames']
        self.contextLength      = cfg['memoryLength']

    def get_ratio_mask(self, o_real, o_imag):
        mag = torch.sqrt(o_real ** 2 + o_imag ** 2)
        phase = torch.atan2(o_imag, o_real)
        mag = torch.tanh(mag)
        return mag, phase

    def apply_mask(self, xr, xi, mag, phase):
        mag = mag * torch.sqrt(xr ** 2 + xi ** 2)
        phase = phase + torch.atan2(xi, xr)
        return mag * torch.cos(phase), mag * torch.sin(phase)

    def forward(self, xr, xi, runFlag):

        if runFlag:
            input_real, input_imag = xr, xi
            skips = list()
            for encoder in self.encoders:
                xr, xi = encoder(xr, xi)
                skips.append((xr, xi))

            xr, xi = self.gruLayer(xr, xi)

            skip = skips.pop()
            skip = None
            for decoder in self.decoders:
                xr, xi = decoder(xr, xi, skip)
                skip = skips.pop()

            xr, xi = padded_cat(xr, skip[0], dim=1), padded_cat(xi, skip[1],
                                                                dim=1)  # ensures skip connection sizes are compatible
            xr, xi = self.final_dec(xr, xi)

            input_real = input_real[:, 0, :, :].unsqueeze(dim=1)
            input_imag = input_imag[:, 0, :, :].unsqueeze(dim=1)

            xr, xi = pad2d_as(xr, input_real), pad2d_as(xi, input_imag)
            mag_mask, phase_corr = self.get_ratio_mask(xr, xi)

            return self.apply_mask(input_real, input_imag, mag_mask, phase_corr)
        else:
            for encoder in self.encoders:
                xr, xi = encoder(xr, xi)
            xr, xi = self.gruLayer(xr, xi)
            return 1

    def training_step(self, batch, batch_idx):
        farend, residual, source = batch

        batchSize = len(farend[:, 0, 0])

        self.gruLayer.reset_internal_stat(batchSize=batchSize, Device=self.device)

        farend_f    = stft(farend.squeeze()).unsqueeze(dim=1)
        residual_f  = stft(residual.squeeze()).unsqueeze(dim=1)
        source_f    = stft(source.squeeze()).unsqueeze(dim=1)

        residual_f_r, farend_f_r = residual_f[..., 1:-1, 0], farend_f[..., 1:-1, 0]
        residual_f_i, farend_f_i = residual_f[..., 1:-1, 1], farend_f[..., 1:-1, 1]
        source_f_r, source_f_i   = source_f[..., 1:-1, 0], source_f[..., 1:-1, 1]

        for memoryInd in range(1, self.contextLength):
            xr, xi = torch.cat((residual_f_r[..., memoryInd:memoryInd+2], farend_f_r[..., memoryInd:memoryInd+2]),
                               dim=1), torch.cat((residual_f_i[..., memoryInd:memoryInd+2], farend_f_i[..., memoryInd:memoryInd+2]), dim=1)
            _ = self.forward(xr, xi, runFlag=False)

        xr, xi = torch.cat((residual_f_r[..., self.contextLength:self.contextLength+2], farend_f_r[..., self.contextLength:self.contextLength + 2]),
                           dim=1), torch.cat((residual_f_i[..., self.contextLength:self.contextLength+2],
                                              farend_f_i[..., self.contextLength:self.contextLength + 2]), dim=1)
        out_r, out_i    = self.forward(xr, xi, runFlag=1)
        out_r, out_i = torch.squeeze(out_r, 1), torch.squeeze(out_i, 1)
        out_audio = istft(out_r, out_i)
        out_audio = out_audio[:, :, -2*self.hop_length:-1*self.hop_length].squeeze() # take only last two frames

        residual = istft(residual_f_r[..., self.contextLength:self.contextLength+2].squeeze(dim=1), residual_f_i[..., self.contextLength:self.contextLength+2].squeeze(dim=1)).squeeze()
        source = istft(source_f_r[..., self.contextLength:self.contextLength+2].squeeze(dim=1), source_f_i[..., self.contextLength:self.contextLength+2].squeeze(dim=1)).squeeze()

        loss = utils.wSDRLoss(residual[:, -2*self.hop_length:-1*self.hop_length].squeeze(),
                              source[:, -2*self.hop_length:-1*self.hop_length].squeeze(), out_audio.squeeze(), 2e-8)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        farend, residual, source, _ = batch

        farend_f    = stft(farend.squeeze()).unsqueeze(dim=1)
        residual_f  = stft(residual.squeeze()).unsqueeze(dim=1)
        source_f    = stft(source.squeeze()).unsqueeze(dim=1)

        batchSize = len(farend[:, 0, 0])
        self.gruLayer.reset_internal_stat(batchSize=batchSize, Device=self.device)

        residual_f_r, farend_f_r = residual_f[..., 1:-1, 0], farend_f[..., 1:-1, 0]
        residual_f_i, farend_f_i = residual_f[..., 1:-1, 1], farend_f[..., 1:-1, 1]

        nr_frames = len(residual_f[0, 0, 0, :, 0])

        xr, xi = torch.cat((residual_f_r[..., 0:0 + 2], farend_f_r[..., 0:0 + 2]), dim=1), \
                 torch.cat((residual_f_i[..., 0:0 + 2], farend_f_i[..., 0:0 + 2]), dim=1)

        out_r, out_i = self(xr, xi, runFlag=1)
        out_r, out_i = torch.squeeze(out_r, 1), torch.squeeze(out_i, 1)
        out_audio = istft(out_r, out_i).squeeze()
        out_audio = out_audio[:, -2*self.hop_length:-1*self.hop_length].squeeze()

        for frameInd in range(1, nr_frames-3):
            xr, xi = torch.cat((residual_f_r[..., frameInd:frameInd+2], farend_f_r[..., frameInd:frameInd+2]), dim=1), \
                     torch.cat((residual_f_i[..., frameInd:frameInd+2], farend_f_i[..., frameInd:frameInd+2]), dim=1)
            out_r, out_i    = self(xr, xi, runFlag=1)
            out_r, out_i    = torch.squeeze(out_r, 1), torch.squeeze(out_i, 1)
            out_audio_tmp   = istft(out_r, out_i).squeeze()
            out_audio_tmp   = out_audio_tmp[:, -2*self.hop_length:-1*self.hop_length].squeeze()
            out_audio       = torch.cat((out_audio, out_audio_tmp), dim=1)

        source = istft(source_f[..., 1:-1, 0].squeeze(dim=1), source_f[..., 1:-1, 1].squeeze(dim=1)).squeeze()
        residual = istft(residual_f_r.squeeze(dim=1), residual_f_i.squeeze(dim=1)).squeeze()

        L = np.minimum(len(out_audio[0, :]), len(source[0, :]))

        loss = utils.wSDRLoss(residual[:, :L].squeeze(), source[:, :L].squeeze(), out_audio[:, :L].squeeze(), 2e-8)


        self.log('val_loss', loss.detach(), on_step=False, on_epoch=True, prog_bar=False)

    def test_step(self, batch, batch_idx):
        try:
            farend, residual, source, item_name = batch
        except:
            farend, residual, item_name = batch

        batchSize = len(farend[:, 0, 0])
        self.gruLayer.reset_internal_stat(batchSize=batchSize, Device=self.device)

        farend_f = stft(farend.squeeze()).unsqueeze(dim=1)
        residual_f = stft(residual.squeeze()).unsqueeze(dim=1)

        residual_f_r, farend_f_r = residual_f[..., 1:-1, 0], farend_f[..., 1:-1, 0]
        residual_f_i, farend_f_i = residual_f[..., 1:-1, 1], farend_f[..., 1:-1, 1]

        nr_frames = len(residual_f[0, 0, 0, :, 0])

        xr, xi = torch.cat((residual_f_r[..., 0:0 + 2], farend_f_r[..., 0:0 + 2]), dim=1), \
                 torch.cat((residual_f_i[..., 0:0 + 2], farend_f_i[..., 0:0 + 2]), dim=1)
        out_r, out_i = self(xr, xi, runFlag=1)
        out_r, out_i = torch.squeeze(out_r, 1), torch.squeeze(out_i, 1)
        out_audio = istft(out_r, out_i).squeeze()
        out_audio = out_audio[:, -2 * self.hop_length:-1 * self.hop_length].squeeze()

        for frameInd in range(1, nr_frames - 3):
            xr, xi = torch.cat((residual_f_r[..., frameInd:frameInd + 2], farend_f_r[..., frameInd:frameInd + 2]),
                               dim=1), \
                     torch.cat((residual_f_i[..., frameInd:frameInd + 2], farend_f_i[..., frameInd:frameInd + 2]),
                               dim=1)
            out_r, out_i = self(xr, xi, runFlag=1)
            out_r, out_i = torch.squeeze(out_r, 1), torch.squeeze(out_i, 1)
            out_audio_tmp = istft(out_r, out_i).squeeze()
            out_audio_tmp = out_audio_tmp[:, -2 * self.hop_length:-1 * self.hop_length].squeeze()
            out_audio = torch.cat((out_audio, out_audio_tmp), dim=1)

        nrSamples = len(item_name)
        for sampleId in range(0, nrSamples):
            s_hat    = out_audio[sampleId, :]
            file     = item_name[sampleId]
            sig= {}
            sig['s_hat']            = s_hat.detach().cpu().numpy()
            scipy.io.savemat('./results/'+file+'.mat', sig)


    def configure_optimizers(self):
        optimizer= Adam(self.parameters(), lr=1e-3) # 6e-4
        lrLambda = lambda epoch: 0.95**epoch
        scheduler= lrSched.LambdaLR(optimizer, lrLambda)
        return [optimizer], [scheduler]









