import React from 'react';
import Link from 'next/link';

export const Toolbar: React.FC = () => {

  return (
    <div className="toolbar" style={{ position: 'fixed', zIndex: 100, justifyContent: 'space-between'}}>
      <Link href="/" className="flex items-center font-bold">
        <img src="/Phantom.svg" alt="Third Space Logo" className="h-10" />
      </Link>
      <div className="flex justify-end space-x-4">
        
      </div>
    </div>
  );
};

export default Toolbar;