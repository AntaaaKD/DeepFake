// Sélectionne la zone d'upload et l'input file
const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file');

// Empêche le comportement par défaut (ouvrir le fichier dans le navigateur) pour le glisser-déposer
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
  uploadArea.addEventListener(eventName, preventDefaults, false);
  document.body.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
  e.preventDefault();
  e.stopPropagation();
}

// Ajoute une classe CSS lors du survol de la zone d'upload
['dragenter', 'dragover'].forEach(eventName => {
  uploadArea.addEventListener(eventName, () => uploadArea.classList.add('highlight'), false);
});

['dragleave', 'drop'].forEach(eventName => {
  uploadArea.addEventListener(eventName, () => uploadArea.classList.remove('highlight'), false);
});

// Lorsque des fichiers sont déposés dans la zone, les affecter à l'input file
uploadArea.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
  const dt = e.dataTransfer;
  const files = dt.files;
  if(files.length > 0) {
    fileInput.files = files;
  }
}
