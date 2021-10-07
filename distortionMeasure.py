import numpy as np
def energyDistortion(ref_wav, in_wav, windowsize, shift):
    if len(ref_wav) == len(in_wav):
        pass
    else:
        print('音频的长度不相等!')
        minlenth = min(len(ref_wav), len(in_wav))
        ref_wav = ref_wav[: minlenth]
        in_wav = in_wav[: minlenth]
    # 每帧语音中有重叠部分，除了重叠部分都是帧移，overlap=windowsize-shift
    # num_frame = (len(ref_wav)-overlap) // shift
    # num_frame = (len(ref_wav)-windowsize+shift) // shift
    num_frame = (len(ref_wav) - windowsize) // shift + 1  # 计算帧的数量

    Distortion = np.zeros(num_frame)
    # 计算每一帧的信噪比
    for i in range(0, num_frame):
        noise_frame_energy = np.sum(ref_wav[i * shift:i * shift + windowsize] ** 2)  # 每一帧噪声的功率
        speech_frame_energy = np.sum(in_wav[i * shift:i * shift + windowsize] ** 2)  # 每一帧信号的功率
        if noise_frame_energy!=0:
            Distortion[i] = np.log10(speech_frame_energy / noise_frame_energy)
        # print(noise_frame_energy,speech_frame_energy,Distortion[i])
        # Distortion[i] = np.abs(np.log10(noise_frame_energy/speech_frame_energy))
    return np.mean(Distortion)

def batchDistortion(origin=None,advs=None,segSize=30,shift=20):
    sum=0
    for id in range(len(origin)):
        # print(origin[id][0])
        sum+=energyDistortion((advs[id][0]-origin[id][0]).numpy(),origin[id][0].numpy(),segSize,shift)
    return float(sum/len(origin))


