#!/bin/bash

# Define colors
BOLD="\033[1m"
YELLOW="\033[33m"
CYAN="\033[36m"
RESET="\033[0m"
GREEN="\033[32m"
DARK_GREY='\033[90m'   # Darker Grey

# Give the required action text some actual content and color
REQUIRED_ACTION="${BOLD}${CYAN}[ACTION REQUIRED] Please press Enter to continue...${RESET}"

printf "
Setting up requirements for DUET.

${BOLD}${YELLOW}Note:${RESET} You will need to take action at various points during this installation.
Before each step that requires your input, a prompt will appear on screen.

Here is an example of what the prompt will look like 👇🏽

${REQUIRED_ACTION}

"
# STEPS
printf "
${BOLD}DUET setup will require the following steps.${RESET}
"
echo ""

STEP_ONE="Install all dependencies for DUET"
STEP_TWO="Generate local config files from example templates"
STEP_THREE="Set up your Modal workspace"
STEP_FOUR="Verify the Modal GPU container and image"
STEP_FIVE="Next steps..."
STEPS=("$STEP_ONE" "$STEP_TWO" "$STEP_THREE" "$STEP_FOUR" "$STEP_FIVE")
count=1
for item in "${STEPS[@]}"
do 
	echo "   $count. $item"
	((count++))
done
    

# Executing Step One.

printf "
${BOLD}${GREEN} STAGE ONE: ${STEP_ONE}.. $RESET

"
echo ""
echo "Checking for an active virtual environment..."

virtual_env_status="Not Active"

if [ -n "$VIRTUAL_ENV" ]; then
	virtual_env_status="Active"
else
	virtual_env_status="Not Active"
fi


if [ "$virtual_env_status" == "Not Active" ]; then
	echo "    ❌ No active virtual environment detected."
	echo "    Please activate your virtual environment and re-run this script."
	exit 1
else
	echo "    ✔️ Virtual environment detected. Proceeding with installation."
fi


# Check which tool to use and install
if command -v uv &> /dev/null; then
	echo "    Installing libraries with uv"
	uv sync || {
		echo ""
		echo "    ❌ ERROR: 'uv sync' failed!"
		echo "    Please ensure your pyproject.toml is valid and your environment is properly set up."
		exit 1
	    }
elif command -v pip3 &> /dev/null; then
	echo "    Installing libraries with pip3"
	pip3 install . || {
		echo ""
		echo "    ❌ ERROR: 'pip3 install' failed!"
		echo "    Note: If you got an 'externally-managed-environment' error, your"
		echo "    virtual environment might not be fully active or recognized by pip3."
		echo "    Try running: source .venv/bin/activate"
		echo "    You can re-run after confirming pip3 is recognized "
		exit 1
	    }
elif command -v pip &> /dev/null; then
	echo "    Installing libraries with pip"
	pip install . || {
		echo ""
		echo "     ❌ ERROR: 'pip install' failed!"
		echo ""
	}
else
    echo "    ❌ Error: Neither uv, pip3, nor pip could be found!"
    exit 1
fi

echo "    Library installation complete."




# ----------------  YAML FILE CREATIONS (STEP 2) --------------------------
printf "

$BOLD$GREEN STAGE TWO: ${STEP_TWO}.. $RESET
This will copy ${DARK_GREY}infra.yaml${RESET} and ${DARK_GREY}image.yaml${RESET} into ${DARK_GREY}config/${RESET} from their example templates. Both files are required to build the DUET container.

$REQUIRED_ACTION

"
read user_input
if [ -n "$user_input" ]; then
    echo ""
    echo "❌ Operation aborted by user."
    exit 1
fi

cp ./config/image.example.yaml ./config/image.yaml && printf "✔️ Successfully created ${DARK_GREY}image.yaml${RESET}\n" || {
	printf "Creating $DARK_GREY image.yaml $RESET failed."
	printf "$DARK_GREY image.yaml $RESET is needed to instantiate the duet container. Without it, container creation will fail"
	exit 1
}
cp ./config/infra.example.yaml ./config/infra.yaml && printf "✔️ Successfully created ${DARK_GREY}infra.yaml${RESET}\n" || {
	printf "Creating $DARK_GREY infra.yaml $RESET failed."
	printf "$DARK_GREY infra.yaml $RESET is needed to instantiate the duet container. Without it, container creation will fail"
	exit 1
}
echo ""


# ----------------  MODAL LOGIN (STEP 3) --------------------------
printf "

$BOLD$GREEN STAGE THREE: ${STEP_THREE}.. $RESET
"

printf "  i.$BOLD Checking if Modal is installed... $RESET"
echo ""
python3 -c "import modal" 2>/dev/null && printf "  ✔️ Modal is installed" || {
	printf "
	❌ $DARK_GREY Modal $RESET is not installed.
	Run 'uv add modal' or 'pip install modal' and re-run this script.
	"
	exit 1
}

printf "

$BOLD Log into your Modal account and select $DARK_GREY duet $RESET as your workspace $RESET

$REQUIRED_ACTION
"
read user_input
if [ -n "$user_input" ]; then
    echo ""
    echo "❌ Operation aborted by user."
    exit 1
fi

modal setup || {
	echo "❌ Modal setup failed. Exiting."
	exit 1
}

printf "

$GREEN Modal setup complete $RESET 🎊
"
echo ""


# ----------------  MODAL LOGIN (STEP 4) --------------------------
printf "

$BOLD$GREEN STAGE FOUR: ${STEP_FOUR}.. $RESET

$BOLD This will spin up a Modal GPU container and verify the DUET image. $RESET

$REQUIRED_ACTION
"


read user_input
if [ -n "$user_input" ]; then
    echo ""
    echo "❌ Operation aborted by user."
    exit 1
fi


modal run scripts/verify_image.py || {
	echo "❌ Container verification failed. Exiting."
	exit 1
}

printf "

$GREEN Container verification complete $RESET 🎊
"
echo ""



# ----------------  NEXT STEPS (STEP 5) --------------------------
printf "

$BOLD$GREEN STAGE FIVE: ${STEP_FIVE}.. $RESET

Setup complete! Here is how to contribute to the repository:
	1. Create a new branch -> $DARK_GREY git switch -c <branch-name> $RESET
	2. Make your changes.
	3. Stage your changes -> $DARK_GREY git add . or git add /path/to/file-or-folder $RESET
	4. Commit your changes -> $DARK_GREY git commit -m 'Your commit message' $RESET
	5. Push to your branch -> $DARK_GREY git push -u origin <branch-name> $RESET
	6. Open a pull request on GitHub for your changes to be reviewed and merged into $DARK_GREY main $RESET

$YELLOW Warning: Do not push directly to the $DARK_GREY main $RESET branch as it is protected. $RESET

"

echo ""


exit
