import React from 'react';
import Link from 'next/link';

export const Toolbar: React.FC = () => {

  return (
    <div className="toolbar" style={{ position: 'fixed', zIndex: 100, justifyContent: 'space-between', }}>
      <Link href="/" className="flex items-center font-bold">
        Insert Logo Here
      </Link>
      <div className="flex justify-end space-x-4">
        <Link href="/foryou"><h3 className="font-bold">Explore</h3></Link>
      </div>
    </div>
  );
};

export default Toolbar;