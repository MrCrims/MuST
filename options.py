class TrainingOptions:
    """
    Configuration options for the training
    """
    def __init__(self,
                 batch_size: int,
                 number_of_epochs: int,
                background_folder:str,  train_folder: str, validation_folder: str, runs_folder: str,
                 start_epoch: int, experiment_name: str):
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs
        self.background_folder = background_folder
        self.train_folder = train_folder
        self.validation_folder = validation_folder
        self.runs_folder = runs_folder
        self.start_epoch = start_epoch
        self.experiment_name = experiment_name

class FWM_Options:
    def __init__(self, H: int, W: int, message_length:int , 
                        encoder_blocks:int , encoder_channels:int,
                        decoder_blocks:int , decoder_channels:int,
                        use_discriminator:bool,
                        discriminator_blocks:int , discriminator_channels:int,
                        decoder_loss:float,
                        encoder_loss:float,
                        adversarial_loss:float,
                        localization_loss:float,
                        resize_bound:list,
                        batchsize:int,
                        enable_fp16:bool = False   ) -> None:
        self.H = H
        self.W = W
        self.message_length = message_length
        self.encoder_blocks = encoder_blocks
        self.encoder_channels = encoder_channels
        self.decoder_blocks = decoder_blocks
        self.decoder_channels = decoder_channels
        self.use_discriminator = use_discriminator
        self.discriminator_blocks = discriminator_blocks
        self.discriminator_channels = discriminator_channels
        self.decoder_loss = decoder_loss
        self.encoder_loss = encoder_loss
        self.adversarial_loss = adversarial_loss
        self.localization_loss = localization_loss
        self.enable_fp16 = enable_fp16
        self.resize_bound = resize_bound
        self.batchsize = batchsize