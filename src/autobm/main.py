from .data.creation import create_dataloaders
from .autobm import AutoBM
from .client import LLMClient
from .prompt.manager import PromptManager

def main():
    cl = LLMClient()
    pm = PromptManager()
    train_loader, valid_loader, test_loader = create_dataloaders()
    # Initialize the AutoBM class
    abm = AutoBM(
        client=cl,
        pm = pm,
        train = train_loader,
        val = valid_loader,
        test = test_loader
    )
    # Run the AutoBM pipeline
    abm.start()

if __name__ == "__main__":
    main()