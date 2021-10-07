import MyAttackModel as ta

class attack():
    def get(self,name,model,params):
        atk=None
        if name == "fgsm":
            atk = ta.FGSM(model, eps=params['fgsm_eps'])
        elif name == "pgd":
            atk = ta.PGD(model, eps=params['pgd_eps'],alpha=params['pgd_alpha'], steps=params['pgd_steps'],random_start=params['pgd_random_start'])
        # elif name == "deepfool":
        #     atk=ta.DeepFool(model,steps=params['deepfool_steps'],overshoot=params['deepfool_overshoot'])
        # elif name == "cw":
        #     atk=ta.CW(model,c=params['cw_c'], kappa=params['cw_kappa'],steps=params['cw_steps'],lr=params['cw_lr'])
        else:
            raise Exception('no such attackMethod choice')
        return atk




