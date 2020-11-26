# source this file!
if [[ "${BASH_SOURCE[0]}" == "${0}" ]] ; then
    echo "ERROR: this file is only sourced by other venv-* files!"
    exit 1
fi

# load settings for this package
source package.conf

# get the local side
function get_site() {
    RESULT=$(hostname -d | sed 's/\./ /g' | rev | awk '{ print $1"."$2}' | rev)
    echo $RESULT
}

# prepare the environment for the current site
function setup_environment() {
    # the name of the current site is the last part of the
    # top-level domain with dots replaced by dashes, e.g., uni-muenchen-de.
    site=$(get_site)
    env_site="environment-${site/./-}.sh"
    if [[ ! -f ${env_site} ]] ; then
      echo "INFO: No site-specific settings found (no ${env_site})!"
    else
      source ${env_site}
    fi
}
