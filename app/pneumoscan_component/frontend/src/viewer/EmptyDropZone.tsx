import { useRef, useState } from "react";
import { MAX_UPLOAD_BYTES } from "../contract";

interface Props {
  onChooseFile: (file: File) => void;
  onUseSample: () => void;
  clientError: string | null;
  setClientError: (message: string | null) => void;
}

const ACCEPT = "image/jpeg,image/png,.jpg,.jpeg,.png,.dcm,.dicom";

export function EmptyDropZone({ onChooseFile, onUseSample, clientError, setClientError }: Props) {
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const acceptFile = (file: File | undefined | null) => {
    if (!file) return;
    if (file.size > MAX_UPLOAD_BYTES) {
      setClientError(`This file is ${(file.size / (1024 * 1024)).toFixed(1)}MB, over the 25MB upload limit.`);
      return;
    }
    setClientError(null);
    onChooseFile(file);
  };

  return (
    <div
      className={`psc-dropzone${dragging ? " psc-dropzone--drag" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        acceptFile(e.dataTransfer.files?.[0]);
      }}
    >
      <div className="psc-dropzone-icon" aria-hidden="true">
        <span />
        <span />
      </div>
      <div className="psc-dropzone-copy">
        <div className="psc-dropzone-title">Add a chest X-ray to begin</div>
        <div className="psc-dropzone-sub">JPEG, PNG, or DICOM · a frontal (PA) chest X-ray works best</div>
      </div>
      <div className="psc-dropzone-actions">
        <button type="button" className="psc-btn psc-btn--primary psc-btn--lg" onClick={() => fileInputRef.current?.click()}>
          Choose X-ray file
        </button>
        <button type="button" className="psc-btn psc-btn--ghost-dark psc-btn--lg" onClick={onUseSample}>
          Use sample X-ray
        </button>
      </div>
      {clientError && <div className="psc-dropzone-error">{clientError}</div>}
      <div className="psc-dropzone-attribution">Sample: CC0 radiograph by Mikael Häggström, Wikimedia Commons</div>
      <input
        ref={fileInputRef}
        type="file"
        accept={ACCEPT}
        style={{ display: "none" }}
        onChange={(e) => {
          acceptFile(e.target.files?.[0]);
          e.target.value = "";
        }}
      />
    </div>
  );
}
